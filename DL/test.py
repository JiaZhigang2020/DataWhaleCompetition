import ray
import time

database = [1] * 1024 * 1024 * 4



def retrieve(item):
    time.sleep(item / 10.)
    return item, database[item]

@ray.remote
def retrieve_task(item, index_num):
    index_num = ray.get(index_num)
    index_num += 1
    print(f'Processing task {index_num}')
    [i + 1 for i in range(int(1e6))]
    return item

db_object_ref = ray.put(database)
index_num = ray.put(0)
task_list = [retrieve_task.remote(db_object_ref, index_num) for _ in range(10000)]
a = ray.get(task_list)
print(a)