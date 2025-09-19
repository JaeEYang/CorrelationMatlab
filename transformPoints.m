function transformedPts = transformPoints(M, pts)
% Apply transformation matrix to points
% pts: 3xN point format
transformedPts = M * pts;
end
