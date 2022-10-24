# Localization of Objects by the Vision System

<br>

<p align="justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project integrates a computer vision system for the localization of certain objects according to a couple of algorithms, to mutually compare their results in the terms of reliability, stability and demand of computing power and to evaluate the possibilities of using them on the real industrial set. The system used in this work consists of a camera and a computer which processes the data retrieved from camera. The algorithms for object recognition are implemented in the programming language Python, along with the open-source library OpenCV. The resulting mutual comparison of the algorithms is based on the industrial set of bottle caps.</p>

<br>


## Project workflow

<br>

<b>Step 1.</b>&nbsp;&nbsp;Determining the task
<br>
<p align="center"><img src="images%20for%20GitHub/object%20recognition.jpg" width="540px"></p>
<br>

<b>Step 2.</b>&nbsp;&nbsp;Defining the industrial set of bottle caps
<br>
<p align="center"><img src="images%20for%20GitHub/set%20of%20caps.jpg" width="400px"></p>
<br>

<b>Step 3.</b>&nbsp;&nbsp;Implementing the gray template matching algorithm
<br>
<p align="center"><img src="images%20for%20GitHub/gray%20template%20matching.jpg" width="480px"></p>
<br>

<b>Step 4.</b>&nbsp;&nbsp;Implementing the binary template matching algorithm
<br>
<p align="center"><img src="images%20for%20GitHub/binary%20template%20matching.jpg" width="480px"></p>
<br>

<b>Step 5.</b>&nbsp;&nbsp;Implementing the Canny edges Hough algorithm
<br>
<p align="center"><img src="images%20for%20GitHub/Canny%20edges%20Hough.jpg" width="480px"></p>
<br>

<b>Step 6.</b>&nbsp;&nbsp;Implementing the adaptive threshold Hough algorithm
<br>
<p align="center"><img src="images%20for%20GitHub/adaptive%20threshold%20Hough.jpg" width="480px"></p>
<br>

<b>Step 7.</b>&nbsp;&nbsp;Implementing the Haar cascade algorithm
<br>
<p align="center"><img src="images%20for%20GitHub/Haar%20cascade.jpg" width="480px"></p>
<br>

<b>Step 8.</b>&nbsp;&nbsp;Analyzing the results for different algorithms (shown for Haar cascade algorithm)
<br>
<p align="center"><img src="images%20for%20GitHub/tested%20Haar%20cascade.jpg" width="480px"></p>
<br>


## Run the project on Windows

<br>

<b>Step 1.</b>&nbsp;&nbsp;Clone the repository:
<pre>
cd %HOMEPATH%

git clone https://github.com/Doc1996/object-localization
</pre>
<br>

<b>Step 2.</b>&nbsp;&nbsp;Create the virtual environment and install dependencies:
<pre>
cd %HOMEPATH%\object-localization

python -m pip install --upgrade pip
python -m pip install --user virtualenv

python -m venv python-virtual-environment
.\python-virtual-environment\Scripts\activate

.\WINDOWS_INSTALLING_PACKAGES.bat
</pre>
<br>

<b>Step 3.</b>&nbsp;&nbsp;Modify the changeable variables in <i>OL_constants.py</i>
<br>
<br>

<b>Step 4.</b>&nbsp;&nbsp;Run the program:
<pre>
cd %HOMEPATH%\object-localization

.\python-virtual-environment\Scripts\activate

.\WINDOWS_OBJECT_LOCALIZATION_APPLICATION.bat
</pre>
<br>


## Run the project on Linux

<br>

<b>Step 1.</b>&nbsp;&nbsp;Clone the repository:
<pre>
cd $HOME

git clone https://github.com/Doc1996/object-localization
</pre>
<br>

<b>Step 2.</b>&nbsp;&nbsp;Create the virtual environment and install dependencies:
<pre>
cd $HOME/object-localization

python3 -m pip install --upgrade pip
python3 -m pip install --user virtualenv

python3 -m venv python-virtual-environment
source python-virtual-environment/bin/activate

source LINUX_INSTALLING_PACKAGES.sh
</pre>
<br>

<b>Step 3.</b>&nbsp;&nbsp;Modify the changeable variables in <i>OL_constants.py</i>
<br>
<br>

<b>Step 4.</b>&nbsp;&nbsp;Run the program:
<pre>
cd $HOME/object-localization

source python-virtual-environment/bin/activate

source LINUX_OBJECT_LOCALIZATION_APPLICATION.sh
</pre>
<br>