#image to string
reduce text in the wild problem to the problem recognizing text with tesseract
##dependencies
opencv3.1 with text module
tesseract
pytesseract

##TODO
estimate density of the text region like edgebox
get the region for the text
## thoughts
the good part of tesseract is given text in image, if the text is more like in the scan format, tesseract is pretty good at recognizing all the text correctly and even separate them correctly with the bounding box coordinates.

However, if it's in a natural scene, it's pretty hard for tesseract to work perfectly. Therefore a good text detector or text region detector is important, in this case text region detector is good enough for tesseract.