from time import strftime

class Logger:
	@staticmethod
	def log(header, msg):
		print(header + str(msg))
		
	@staticmethod
	def info(msg):
		header = strftime("[%Y-%m-%d %H:%M:%S] INFO: ")
		Logger.log(header, msg)
	
	@staticmethod
	def warn(msg):
		header = strftime("[%Y-%m-%d %H:%M:%S] WARN: ")
		Logger.log(header, msg)
	
	