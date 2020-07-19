#include <opencv2/opencv.hpp>
#include <raspicam_cv.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <wiringPi.h>

using namespace std;
using namespace cv;
using namespace raspicam;

// Image Processing variables
Mat frame, Matrix, framePers, frameGray, frameThresh, frameEdge, frameFinal, frameFinalDuplicate, frameFinalDuplicate1;
Mat ROI, ROIRowEnd;
int LeftPos, RightPos, frameCenter, Center, Result, RowEnd;

RaspiCam_Cv Camera;

stringstream ss;


vector<int> histrogram;
vector<int> histrogramRowEnd;

Point2f Source[] = {Point2f(40,135),Point2f(360,135),Point2f(0,185), Point2f(400,185)};
Point2f Destination[] = {Point2f(100,0),Point2f(280,0),Point2f(100,240), Point2f(280,240)};


//Machine Learning variables
CascadeClassifier weed_Cascade, crop_Cascade;
Mat frame_weed, RoI_weed, gray_weed, frame_crop, RoI_crop, gray_crop;
vector<Rect> weed, crop;
int dist_weed, dist_crop;

 void Setup ( int argc,char **argv, RaspiCam_Cv &Camera )
  {
    Camera.set ( CAP_PROP_FRAME_WIDTH,  ( "-w",argc,argv,400 ) );
    Camera.set ( CAP_PROP_FRAME_HEIGHT,  ( "-h",argc,argv,240 ) );
    Camera.set ( CAP_PROP_BRIGHTNESS, ( "-br",argc,argv,50 ) );
    Camera.set ( CAP_PROP_CONTRAST ,( "-co",argc,argv,50 ) );
    Camera.set ( CAP_PROP_SATURATION,  ( "-sa",argc,argv,50 ) );
    Camera.set ( CAP_PROP_GAIN,  ( "-g",argc,argv ,50 ) );
    Camera.set ( CAP_PROP_FPS,  ( "-fps",argc,argv,0));

}

void Capture()
{
    Camera.grab();
    Camera.retrieve( frame);
    cvtColor(frame, frame_weed, COLOR_BGR2RGB);
    cvtColor(frame, frame_crop, COLOR_BGR2RGB);
    cvtColor(frame, frame, COLOR_BGR2RGB);
    
}

void Perspective()
{
	line(frame,Source[0], Source[1], Scalar(0,0,255), 2);
	line(frame,Source[1], Source[3], Scalar(0,0,255), 2);
	line(frame,Source[3], Source[2], Scalar(0,0,255), 2);
	line(frame,Source[2], Source[0], Scalar(0,0,255), 2);
	
	
	Matrix = getPerspectiveTransform(Source, Destination);
	warpPerspective(frame, framePers, Matrix, Size(400,240));
}

void Threshold()
{
	cvtColor(framePers, frameGray, COLOR_RGB2GRAY);
	inRange(frameGray, 230, 255, frameThresh);
	Canny(frameGray,frameEdge, 900, 900, 3, false);
	add(frameThresh, frameEdge, frameFinal);
	cvtColor(frameFinal, frameFinal, COLOR_GRAY2RGB);
	cvtColor(frameFinal, frameFinalDuplicate, COLOR_RGB2BGR);   //used in histrogram function only
	cvtColor(frameFinal, frameFinalDuplicate1, COLOR_RGB2BGR);   //used in histrogram function only
	
}

void Histrogram()
{
    histrogram.resize(400);
    histrogram.clear();
    
    for(int i=0; i<400; i++)       //frame.size().width = 400
    {
	ROI= frameFinalDuplicate(Rect(i,140,1,100));
	divide(255, ROI, ROI);
	histrogram.push_back((int)(sum(ROI)[0])); 
    }
	
	histrogramRowEnd.resize(400);
        histrogramRowEnd.clear();
	for (int i = 0; i < 400; i++)       
	{
		ROIRowEnd = frameFinalDuplicate1(Rect(i, 0, 1, 240));   
		divide(255, ROIRowEnd, ROIRowEnd);       
		histrogramRowEnd.push_back((int)(sum(ROIRowEnd)[0]));  
		
	
	}
	   RowEnd = sum(histrogramRowEnd)[0];
	   cout<<"Row END = "<<RowEnd<<endl;
}

void RowFinder()
{
    vector<int>:: iterator LeftPtr;
    LeftPtr = max_element(histrogram.begin(), histrogram.begin() + 150);
    LeftPos = distance(histrogram.begin(), LeftPtr); 
    
    vector<int>:: iterator RightPtr;
    RightPtr = max_element(histrogram.begin() +250, histrogram.end());
    RightPos = distance(histrogram.begin(), RightPtr);
    
    line(frameFinal, Point2f(LeftPos, 0), Point2f(LeftPos, 240), Scalar(0, 255,0), 2);
    line(frameFinal, Point2f(RightPos, 0), Point2f(RightPos, 240), Scalar(0,255,0), 2); 
}

void RowCenter()
{
    Center = (RightPos-LeftPos)/2 +LeftPos;
    frameCenter = 188;
    
    line(frameFinal, Point2f(Center,0), Point2f(Center,240), Scalar(0,255,0), 3);
    line(frameFinal, Point2f(frameCenter,0), Point2f(frameCenter,240), Scalar(255,0,0), 3);

    Result = Center-frameCenter;
}


void weed_detection()
{
    if(!weed_Cascade.load("//home//pi//Desktop//MACHINE LEARNING//weed_cascade.xml"))
    {
	printf("Unable to open stop cascade file");
    }
    
    RoI_weed = frame_weed(Rect(0,0,400,240));
    cvtColor(RoI_weed, gray_weed, COLOR_RGB2GRAY);
    equalizeHist(gray_weed, gray_weed);
    weed_Cascade.detectMultiScale(gray_weed, weed);
    
    for(int i=0; i<weed.size(); i++)
    {
	Point P1(weed[i].x, weed[i].y);
	Point P2(weed[i].x + weed[i].width, weed[i].y + weed[i].height);
	
	rectangle(RoI_weed, P1, P2, Scalar(0, 0, 255), 2);
	putText(RoI_weed, "weed", P1, FONT_HERSHEY_PLAIN, 1,  Scalar(0, 0, 255, 255), 2);
	dist_weed = (-1.07)*(P2.x-P1.x) + 102.597;
	
       ss.str(" ");
       ss.clear();
       ss<<"D = "<<dist_weed<<"cm";
       putText(RoI_weed, ss.str(), Point2f(1,130), 0,1, Scalar(0,0,255), 2);
	
    }
    
}
                 

void crop_detection()
{
    if(!Crop_Cascade.load("//home//pi//Desktop//MACHINE LEARNING//crop_cascade.xml"))
    {
	printf("Unable to open Object cascade file");
    }
    
    RoI_crop = frame_crop(Rect(100,50,200,190));
    cvtColor(RoI_crop, gray_crop, COLOR_RGB2GRAY);
    equalizeHist(gray_crop, gray_crop);
    crop_Cascade.detectMultiScale(gray_crop, crop);
    
    for(int i=0; i<crop.size(); i++)
    {
	Point P1(crop[i].x, crop[i].y);
	Point P2(crop[i].x + crop[i].width, crop[i].y + crop[i].height);
	
	rectangle(RoI_crop, P1, P2, Scalar(0, 0, 255), 2);
	putText(RoI_crop, "Object", P1, FONT_HERSHEY_PLAIN, 1,  Scalar(0, 0, 255, 255), 2);
	dist_crop = (-0.48)*(P2.x-P1.x) + 56.6;
	
       ss.str(" ");
       ss.clear();
       ss<<"D = "<<dist_Object<<"cm";
       putText(RoI_crop, ss.str(), Point2f(1,130), 0,1, Scalar(0,0,255), 2);
	
    }
    
}


int main(int argc,char **argv)
{
	
    wiringPiSetup();
    pinMode(21, OUTPUT);
    pinMode(22, OUTPUT);
    pinMode(23, OUTPUT);
    pinMode(24, OUTPUT);
    
    Setup(argc, argv, Camera);
    cout<<"Connecting to camera"<<endl;
    if (!Camera.open())
    {
		
	cout<<"Failed to Connect"<<endl;
    }
     
	cout<<"Camera Id = "<<Camera.getId()<<endl;
     
 
    while(1)
    {
	
    auto start = std::chrono::system_clock::now();
    
    
    Capture();               
    Perspective();
    Threshold();
    Histrogram();
    RowFinder();
    RowCenter();
    weed_detection();
    crop_detection();
    
    
    
    if (dist_weed >0 && dist_weed <60)
    {
	digitalWrite(21, 0);
	digitalWrite(22, 0);    //decimal = 8
	digitalWrite(23, 0);
	digitalWrite(24, 1);
	cout<<"weed"<<endl;
 
	goto weed;
    }
    
       if (dist_crop > 5 && dist_crop < 20)
    {
	digitalWrite(21, 1);
	digitalWrite(22, 0);    //decimal = 9
	digitalWrite(23, 0);
	digitalWrite(24, 1);
	cout<<"Crop"<<endl;
	dist_Crop = 0;
	
	goto crop;
    }
    
 
    
   else if (RowEnd > 3000)
    {
       	digitalWrite(21, 1);
	digitalWrite(22, 1);    //decimal = 7
	digitalWrite(23, 1);
	digitalWrite(24, 0);
	cout<<"row End"<<endl;
    }
    
    
   else if (Result == Result)
    {
	
	digitalWrite(21, 0);
	digitalWrite(22, 0);    //decimal = 0
	digitalWrite(23, 0);
	digitalWrite(24, 0);
	cout<<"Forward"<<endl;
    }
    
        
    
    
    weed:
    crop:
  
   
    
   if (RowEnd > 3000)
    {
       ss.str(" ");
       ss.clear();
       ss<<" row End";
       putText(frame, ss.str(), Point2f(1,50), 0,1, Scalar(255,0,0), 2);
    
     }
    
    else if (Result == Result)
    {
       ss.str(" ");
       ss.clear();
       ss<<" (Move Forward)";
       putText(frame, ss.str(), Point2f(1,50), 0,1, Scalar(0,0,255), 2);
    
     }
    
    
    namedWindow("orignal", WINDOW_KEEPRATIO);
    moveWindow("orignal", 0, 100);
    resizeWindow("orignal", 640, 480);
    imshow("orignal", frame);
    
    namedWindow("Perspective", WINDOW_KEEPRATIO);
    moveWindow("Perspective", 640, 100);
    resizeWindow("Perspective", 640, 480);
    imshow("Perspective", framePers);
    
    namedWindow("Final", WINDOW_KEEPRATIO);
    moveWindow("Final", 1280, 100);
    resizeWindow("Final", 640, 480);
    imshow("Final", frameFinal);
    
    namedWindow("weed", WINDOW_KEEPRATIO);
    moveWindow("weed", 1280, 580);
    resizeWindow("weed", 640, 480);
    imshow("weed", RoI_weed);
    
    namedWindow("crop", WINDOW_KEEPRATIO);
    moveWindow("crop", 640, 580);
    resizeWindow("crop", 640, 480);
    imshow("crop", RoI_crop);
  
    waitKey(1);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    
    float t = elapsed_seconds.count();
    int FPS = 1/t;
    //cout<<"FPS = "<<FPS<<endl;
    
    }

    
    return 0;
     
}

