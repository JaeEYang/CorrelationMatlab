function pts = loadCSV(filename)
% loadCSV: Load CSV points and ensure 3xN format
% pts = loadCSV(filename)
pts = readmatrix(filename);
if size(pts,1) ~= 3 && size(pts,2) == 3
   pts = pts';
end
end
