
Graph-cut Image Segmentation
----------------------------

Implements Boykov/Kolmogorov�s Max-Flow/Min-Cut algorithm for computer vision problems. Two
gray-scale images have been used to test the system for image segmentation (foreground/background segmentation) problem. 

Steps:
1. defined the graph structure and unary and pairwise terms. For graph structure, i have used available
packages/libraries such as PyMaxflow.
2. likelihood function for background and foreground has been generated.
3. General energy function consisting of unary and pairwise energy functionals have been written.
4. Likelihood maps (intensity map ranging from 0 to 1) for foreground and background have been displayed.
5. Use Boykov/Kolmogorov maxflow / mincut approach for solving the energy minimization problem.
6. Final segmentation have been displayed. Created an image for which the background pixels are red, and the foreground pixels have the color of the input image.

Relevant paper can be found here: http://www.csd.uwo.ca/~yuri/Papers/pami04.pdf