import os
import zipfile


# caffe
caffe_zip_file_name = "caffe-windows-09122016.zip"
caffe_zip = zipfile.ZipFile(caffe_zip_file_name)
#caffe_zip.extractall()
caffe_dep_download_file = "./caffe-windows/scripts/download_prebuilt_dependencies.py"
os.system('python ' + caffe_dep_download_file + ' --msvc_version=v140')