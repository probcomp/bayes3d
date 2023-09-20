import paramiko
from scp import SCPClient

class SSHSender:
    def __init__(self, hostname, username, ssh_key_path, result_directory):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname, username=username, key_filename=ssh_key_path)
        self.scp = SCPClient(ssh.get_transport())

        self.result_directory = result_directory

    def send_file(self, filename):
        self.scp.put(filename, remote_path=self.result_directory)
