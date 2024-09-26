from flask import Flask
from flask_apscheduler import APScheduler
from drl_task import run_drl_task  # Ensure this function is defined in drl_task.py

class Config:
    SCHEDULER_API_ENABLED = True

app = Flask(__name__)
app.config.from_object(Config())

scheduler = APScheduler()

def scheduled_task():
    run_drl_task()
    print("++++++++++run+++++++++")

@app.route('/')
def index():
    return 'DRL Task Scheduler is running.'

if __name__ == '__main__':
    # scheduler.init_app(app)
    # Schedule the task to run every 10 seconds
    scheduler.add_job(id='Scheduled Task', func=scheduled_task, trigger='interval', seconds=10)
    scheduler.start()
    app.run(debug=True)
