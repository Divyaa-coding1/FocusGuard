import subprocess

def send_notification(title, message):
    script = f'''
    display notification "{message}" with title "{title}"
    '''
    subprocess.run(["osascript", "-e", script])
