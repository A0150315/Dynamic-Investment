import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import logging

def send_log_by_email(log_file_path, delete_after=True):
    """
    将日志文件通过邮件发送，并可选择发送后删除日志文件
    
    Args:
        log_file_path (str): 日志文件的完整路径
        delete_after (bool): 发送后是否删除本地日志文件
        
    Returns:
        bool: 发送成功返回True，否则返回False
    """
    try:
        # 邮件设置
        smtp_server = "smtp.163.com"
        port = 25
        sender_email = "tjqtest@163.com"
        sender_name = "Investment"
        password = "ZEUR238ZatiUWJ3Y"
        receiver_email = "574469551@qq.com"
        
        # 检查日志文件是否存在
        if not os.path.exists(log_file_path):
            logging.error(f"日志文件 {log_file_path} 不存在，无法发送")
            return False
            
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = f"{sender_name} <{sender_email}>"
        msg['To'] = receiver_email
        
        # 添加日期到主题
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        msg['Subject'] = f"股票回测日志 - {today}"
        
        # 邮件正文
        body = f"""
        这是 {today} 的股票回测日志文件。
        
        包含了今天运行的所有股票回测建议。
        
        自动发送，请勿回复。
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # 添加日志文件作为附件
        with open(log_file_path, 'rb') as f:
            attachment = MIMEApplication(f.read(), Name=os.path.basename(log_file_path))
            attachment['Content-Disposition'] = f'attachment; filename="{os.path.basename(log_file_path)}"'
            msg.attach(attachment)
        
        # 发送邮件
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()  # 启用 TLS 加密
            server.login(sender_email, password)
            server.send_message(msg)
            logging.info(f"日志已成功发送到 {receiver_email}")
            print(f"日志已成功发送到 {receiver_email}")
        
        # 如果设置了删除，则删除本地日志文件
        if delete_after and os.path.exists(log_file_path):
            os.remove(log_file_path)
            logging.info(f"本地日志文件 {log_file_path} 已删除")
            print(f"本地日志文件 {log_file_path} 已删除")
            
        return True
        
    except Exception as e:
        logging.error(f"发送日志邮件时出错: {e}")
        print(f"发送日志邮件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
		log_file_path = "backtest.log"
		with open(log_file_path, 'w') as f:
			f.write("This is a test log file.")
		send_log_by_email(log_file_path, delete_after=True)