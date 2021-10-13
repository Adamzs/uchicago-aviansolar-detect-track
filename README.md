# uchicago-aviansolar

Requires OpenCV 3.4.2 (but could be made to work with 4.0 if one or two things were updated)<br>
<pre>
usage: main_app_single_file.py [-h] [-show_video] [-write_video] [-write_images] [-v] video_file output_dir

positional arguments:
  video_file     The video file to process
  output_dir     The path to the output folder to use if writing out moving object images

optional arguments:
  -h, --help     show this help message and exit
  -show_video    Should the video be displayed while processing?
  -write_video   Should video output with detection boxes be written to disk?
  -write_images  Should images of the moving objects be written to disk?
  -v             Verbose - should more detail be written about progress?</pre>
<br>
example: <br>
<code>python main_app_single_file.py C:\Users\szymanski\aviansolar-testdata\example1\video-00104-2020_04_26_11_57_51.mkv C:\Users\szymanski\aviansolar-testdata\example1 -show_video -write_images -write_video -v</code>
