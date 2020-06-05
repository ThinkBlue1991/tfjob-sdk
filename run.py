import copy
import datetime
import os
import six

from kubernetes import client, config

config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

config.load_kube_config(config_file=config_file)

configuration = client.Configuration()

core_v1_api = client.CoreV1Api(client.ApiClient(configuration))

custom_obj_api = client.CustomObjectsApi(client.ApiClient(configuration))

batch_v1_api = client.BatchV1Api(client.ApiClient(configuration))

PRIMITIVE_TYPES = (float, bool, bytes, six.text_type) + six.integer_types


def sanitize_for_serialization(obj):
    if obj is None:
        return None
    elif isinstance(obj, PRIMITIVE_TYPES):
        return obj
    elif isinstance(obj, list):
        return [sanitize_for_serialization(sub_obj)
                for sub_obj in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_serialization(sub_obj)
                     for sub_obj in obj)
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        obj_dict = obj
    else:
        obj_dict = {obj.attribute_map[attr]: getattr(obj, attr)
                    for attr, _ in six.iteritems(obj.openapi_types)
                    if getattr(obj, attr) is not None}

    return {key: sanitize_for_serialization(val)
            for key, val in six.iteritems(obj_dict)}


def get_job_container(c_volume_mounts, c_image, c_resouces, name, c_command):
    volume_mounts = list()

    for key in c_volume_mounts.keys():
        if key == "dataset-dir":
            volume_mount = client.V1VolumeMount(
                mount_path=os.path.realpath(c_volume_mounts[key].strip()),
                name=key, read_only=True)
        else:
            volume_mount = client.V1VolumeMount(
                mount_path=os.path.realpath(c_volume_mounts[key].strip()),
                name=key)

        volume_mounts.append(volume_mount)

    if 'work-dir' in c_volume_mounts.keys():
        working_dir = c_volume_mounts['work-dir']
    else:
        working_dir = None

    return client.V1Container(image=c_image, name=name,
                              resources=c_resouces, command=c_command,
                              volume_mounts=volume_mounts,
                              working_dir=working_dir,
                              image_pull_policy="Always")


def get_job_volumes(volumes):
    volume_list = list()

    for key in volumes.keys():
        host_path = client.V1HostPathVolumeSource(
            path=os.path.realpath(volumes[key].strip()),
            type='DirectoryOrCreate')
        volume = client.V1Volume(name=key, host_path=host_path)

        volume_list.append(volume)

    return volume_list


def get_pod_template(name, c_labels, c_volume_mounts, c_image, c_resources,
                     c_command, volumes):
    annotations = {"sidecar.istio.io/inject": "false"}
    tmp_metadata = client.V1ObjectMeta(annotations=annotations, labels=c_labels)

    containers = list()
    container = get_job_container(c_volume_mounts=c_volume_mounts,
                                  c_image=c_image,
                                  c_resouces=c_resources, name=name,
                                  c_command=c_command)
    containers.append(container)

    volume_list = get_job_volumes(volumes)

    security_context = client.V1PodSecurityContext(fs_group=100, run_as_user=0)
    pod_spec = client.V1PodSpec(containers=containers,
                                restart_policy="Never",
                                security_context=security_context,
                                volumes=volume_list)

    return client.V1PodTemplateSpec(metadata=tmp_metadata, spec=pod_spec)


def get_resources(resources_quota):
    resources = dict()
    resources['requests'] = dict()
    resources['limits'] = dict()
    for key in resources_quota.keys():
        if key == "cpu":
            resources['requests']['cpu'] = str(resources_quota[key])
            resources['limits']['cpu'] = str(resources_quota[key])
        if key == "mem":
            resources['requests']['memory'] = str(resources_quota[key]) + 'Gi'
            resources['limits']['memory'] = str(resources_quota[key]) + 'Gi'
        if key == "gpu":
            resources['requests']['nvidia.com/gpu'] = str(
                resources_quota['gpu'])
            resources['limits']['nvidia.com/gpu'] = str(
                resources_quota['gpu'])

    return resources


def delete_train_tf_job(name, namespace):
    group = "kubeflow.org"
    version = "v1"
    plural = "tfjobs"
    custom_obj_api.delete_namespaced_custom_object(group=group,
                                                   version=version,
                                                   plural=plural,
                                                   namespace=namespace,
                                                   name=name,
                                                   body=client.V1DeleteOptions(
                                                       propagation_policy='Foreground',
                                                       grace_period_seconds=5))


def get_train_tf_job(name, namespace):
    try:
        flag = True
        group = "kubeflow.org"
        version = "v1"
        plural = "tfjobs"
        custom_obj_api.get_namespaced_custom_object(group=group,
                                                    version=version,
                                                    plural=plural,
                                                    namespace=namespace,
                                                    name=name)
    except Exception as ex:
        print(ex)
        flag = False
    finally:
        return flag


def create_train_tf_job(name, namespace, c_labels, c_volume_mounts, c_image,
                        c_command, volumes, chief_resource_quota=None,
                        ps_resource_quota=None,
                        worker_resource_quota=None):
    group = "kubeflow.org"
    version = "v1"
    plural = "tfjobs"

    body = dict()
    body['apiVersion'] = '/'.join((group, version))
    body['kind'] = 'TFJob'
    body['metadata'] = dict()
    body['spec'] = dict()

    body['metadata']['name'] = name
    body['metadata']['namespace'] = namespace

    body['spec']['tfReplicaSpecs'] = dict()
    body['spec']['tfReplicaSpecs']['Chief'] = dict()

    # 创建pod的模板信息
    if chief_resource_quota is not None:
        chief_resources = get_resources(chief_resource_quota)
        chief_replicas = chief_resource_quota['replicas']
    else:
        chief_resources = None
        chief_replicas = 1

    chief_template = sanitize_for_serialization(
        get_pod_template(name="tensorflow", c_labels=c_labels,
                         c_volume_mounts=c_volume_mounts,
                         c_image=c_image,
                         c_resources=chief_resources,
                         c_command=c_command,
                         volumes=volumes))

    body['spec']['tfReplicaSpecs']['Chief']['replicas'] = chief_replicas
    body['spec']['tfReplicaSpecs']['Chief']['restartPolicy'] = "Never"
    body['spec']['tfReplicaSpecs']['Chief']['template'] = chief_template

    if ps_resource_quota is not None:
        ps_resources = get_resources(ps_resource_quota)
        ps_replicas = ps_resource_quota['replicas']
    else:
        ps_resources = None
        ps_replicas = 1
    ps_template = copy.deepcopy(chief_template)
    ps_template['spec']['containers'][0]['resources'] = ps_resources

    body['spec']['tfReplicaSpecs']['PS'] = dict()
    body['spec']['tfReplicaSpecs']['PS']['replicas'] = ps_replicas
    body['spec']['tfReplicaSpecs']['PS']['restartPolicy'] = "Never"
    body['spec']['tfReplicaSpecs']['PS']['template'] = ps_template

    if worker_resource_quota is not None:
        worker_resources = get_resources(worker_resource_quota)
        worker_replicas = worker_resource_quota['replicas']
    else:
        worker_resources = None
        worker_replicas = 1
    worker_template = copy.deepcopy(chief_template)
    worker_template['spec']['containers'][0]['resources'] = worker_resources

    body['spec']['tfReplicaSpecs']['Worker'] = dict()
    body['spec']['tfReplicaSpecs']['Worker']['replicas'] = worker_replicas
    body['spec']['tfReplicaSpecs']['Worker']['restartPolicy'] = "Never"
    body['spec']['tfReplicaSpecs']['Worker']['template'] = worker_template

    custom_obj_api.create_namespaced_custom_object(group=group, version=version,
                                                   plural=plural, body=body,
                                                   namespace=namespace)


if __name__ == "__main__":

    # create tfjob
    name = "tfjob-test"
    namespace = "zhangsan"
    c_labels = None
    c_command = ['python', 'mnist.py', '--tf-data-dir=/dataset/',
                 '--tf-model-dir=/data/model',
                 '--tf-export-dir=/data/log', '--tf-train-steps=200',
                 '--tf-batch-size=100', '--tf-learning-rate=0.01']

    c_image = 'myharbor/tensorflow-1.15.2-cpu:latest'

    ps_resource_quota = {'replicas': 1, 'cpu': 1, 'mem': 1, 'gpu': 0}
    chief_resource_quota = {'replicas': 1, 'cpu': 1, 'mem': 1, 'gpu': 0}
    worker_resource_quota = {'replicas': 2, 'cpu': 2, 'mem': 4, 'gpu': 0}
    c_volume_mounts = {'work-dir': '/app/', 'dataset-dir': '/dataset/',
                       'model-dir': '/data/model', 'log-dir': '/data/log'}

    volumes = {'work-dir': '/mnt/mfs/zhangsan/tfjob/',
               'dataset-dir': '/mnt/mfs/zhangsan/tfjob/dataset',
               'model-dir': '/mnt/mfs/zhangsan/tfjob/models/model',
               'log-dir': '/mnt/mfs/zhangsan/tfjob/models/log'}
    if get_train_tf_job(name=name, namespace=namespace):
        delete_train_tf_job(name=name, namespace=namespace)
    else:
        create_train_tf_job(name=name, namespace=namespace, c_labels=c_labels,
                            c_command=c_command, c_image=c_image,
                            c_volume_mounts=c_volume_mounts, volumes=volumes,
                            ps_resource_quota=ps_resource_quota,
                            chief_resource_quota=chief_resource_quota,
                            worker_resource_quota=worker_resource_quota)
