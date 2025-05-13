from celery import Celery
import os


rmq_user = os.getenv('RABBITMQ_DEFAULT_USER')
rmq_passwd = os.getenv('RABBITMQ_DEFAULT_PASS')
celery = Celery(
    'tasks',
    broker=f'pyamqp://{rmq_user}:{rmq_passwd}@caas-rabbitmq',
    backend='redis://caas-redis/0',
)

# celery = Celery('proj',
#              broker='pyamqp://patrick:914lhSgl6UzvmFhQo8dOKtYtpffLyNrGgoFz6qgI@localhost',
#              backend='mongodb://caas-user:FDMxbsdFazhTuYJvjWgzycF4YP0wlJJMLF0ZlKaQ@localhost:27017/celery_results',
#              backend='redis://localhost/0',
#              include=['proj.tasks']
#              )


# app.conf.mongodb_backend_settings = {
#     'database': 'caas',
#     'taskmeta_collection': 'results'
# }

# if __name__ == '__main__':
#     celery.start()