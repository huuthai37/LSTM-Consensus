def server():
	return True

def data_input_path():
	if server():
		return '/content/datalab/'
	else:
		return '/mnt/UCF-11/'

def data_output_path():
	if server():
		return '/content/datalab/'
	else:
return '/mnt/data-11-t5/'
