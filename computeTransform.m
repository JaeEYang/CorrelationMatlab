function M = computeTransform(P, Q)
% Compute 3x3 affine matrix from FLM to TEM
% P and Q are 3xN matrices
M = Q * P' * inv(P * P');
end
