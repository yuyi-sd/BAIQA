LABORATORY FOR IMAGE AND VIDEO ENGINEERING
at The University of Texas at Austin


-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2015 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without
license or royalty fees, to use, copy, modify, and distribute this
database (the images, the results and the source files) and its 
documentation for any purpose, provided that the copyright 
notice in its entirety appear in all copies of this 
database, and the original source of this database, Laboratory for 
Image and Video Engineering (LIVE, http://live.ece.utexas.edu) at the 
University of Texas at Austin (UT Austin, http://www.utexas.edu), 
is acknowledged in any publication that reports research using this database.
The database is to be cited in the bibliography as:

D. Ghadiyaram and A.C. Bovik, "Massive Online Crowdsourced Study of Subjective and Objective Picture Quality," IEEE Transactions on Image Processing, accepted.

D. Ghadiyaram and A.C. Bovik, "LIVE In the Wild Image Quality Challenge Database," Online: http://live.ece.utexas.edu/research/ChallengeDB/index.html, 2015.

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------

Please contact Deepti Ghadiyaram (deeptigp9@gmail.com) if you have any questions.
This investigators on this research were:
Deepti Ghadiyaram (deeptigp9@gmail.com) -- Department of CS at UT Austin.
Dr. Alan C. Bovik (bovik@ece.utexas.edu) -- Department of ECE at UT Austin.

-------------------------------------------------------------------------

The subjective experiment release comes with the following files:

* This readme file containing copyright information and usage information.
* An Images folder with a total of 1,162 images that were used for in the test phase of our online subjective study.
* An Images/trainingImages folder with a total of 7 images ,that were used for training the subjects of our online subjective study.
* A Data folder containing mat files of images, MOS scores, and standard deviation scores.

DETAILS OF THE DATABASE
~~~~~~~~~~~~~~~~~~~~~~~

The LIVE In the Wild Image Quality Challenge Database contains 1,162 images impaired by a wide variety of randomly occurring distortions and genuine capture artifacts that were obtained
using a wide-variety of contemporary mobile camera devices including smartphones and tablets. We gathered numerous "authentically" distorted images taken by many dozens of casual international users, containing diverse distortion types, mixtures, and severities. The images were collected without artificially introducing any distortions beyond those occurring during capture, processing, and storage. 

Since these images are authentically distorted, they usually contain mixtures of multiple impairments that defy categorization into "distortion types." Such images are encountered in the real world and reflect a broad range of difficult to describe (or pigeon-hole) composite image impairments. 


DETAILS OF THE EXPERIMENTS AND PROCESSING OF RAW SCORES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With a goal to gather a large number of human opinion scores, we designed and implemented an online crowdsourcing system which we used to gather more than 350,000 human ratings of image quality from over 8,100 unique subjects which amounts to an average of 175 ratings on each image in the new LIVE Challenge Database. 

Details of the content and design of the database, our crowdsourcing framework, and the very large scale subjective study we conducted on image quality can be found in:

D. Ghadiyaram and A.C. Bovik, "Massive Online Crowdsourced Study of Subjective and Objective Picture Quality," IEEE Transactions on Image Processing, accepted

D. Ghadiyaram and A.C. Bovik, "LIVE In the Wild Image Quality Challenge Database," Online: http://live.ece.utexas.edu/research/ChallengeDB/index.html, 2015.


DETAILS OF FILES PROVIDED IN THIS RELEASE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MATLAB mat files

1. AllImages_release.mat : Contains the names of all the 1169 images that are part of this database (1162 test images + 7 training images). 

2. AllMOS_release.mat : This file has the mean opinion scores (MOS) corresponding to each of the 1169 images. These values are presented in the same order as the images in AllImages_release.mat

3. AllStdDev_release.mat : This file has the standard deviation obtained on the raw opinion scores obtained from a large number of subjects on each image. These values correspond to the images in AllImages_release.mat in the same order.
