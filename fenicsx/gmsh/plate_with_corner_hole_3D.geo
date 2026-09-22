// mesh size

lc = 5e-2;
L = 1;
H = 1;
W = 0.05;
r = 0.3;

// corner points of the plate

Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, H, 0, lc};
Point(4) = {0, H, 0, lc};

// auxiliary points for the corner hole

Point(31) = {L, H - r, 0, lc};
Point(32) = {L - r, H, 0, lc};

// connect points with lines and arc

Line(1) = {1, 2};
Line(2) = {2, 31};
Line(3) = {32, 4};
Line(4) = {4, 1};

Circle(5) = {31, 3, 32};

// use lines to create a closed curve

Curve Loop(1) = {1, 2, 5, 3, 4};

// asign a surface to the closed curve

Plane Surface(1) = {1};
// Recombine Surface(1) = {1};

// extrude

Extrude {0, 0, W} {Surface{1};}

// make the surface and curves physical
// in gmsh: tools > visibility to find out surface ids

Physical Volume("volume") = {1};
Physical Surface("back") = {1};
Physical Surface("bottom") = {15};
Physical Curve("right") = {19}; // symmetry plane next to hole
Physical Curve("top") = {27}; // symmetry plane next to hole
Physical Curve("left") = {31};
Physical Surface("front") = {32};

// mesh

Mesh 3;
Save "plate_with_corner_hole_3D.msh";
