function f = interactiveOverlay(baseImg, overlayImg, transformedPoints)
% interactiveOverlay: Display base image and overlay image with a slider
% and optional transformed points
%
% Inputs:
% baseImg          - TEM image
% overlayImg       - transformed FLM image
% transformedPoints - 3xN matrix of points (optional)
% Developed by Jae Yang, portions of code documentation and clean up
% supported with OpenAI ChatGPT

if nargin < 3
    transformedPoints = [];
end

% Create figure
f = figure('Name','Interactive Overlay','NumberTitle','off','Position',[200 200 900 600]);
ax = axes('Parent',f,'Position',[0.05 0.2 0.9 0.75]);
imshow(baseImg,[],'Parent',ax); hold(ax,'on');

% Display transformed FLM image
hOverlay = imshow(overlayImg,[],'Parent',ax);
hOverlay.AlphaData = 0.5;

% Display transformed points (optional)
if ~isempty(transformedPoints)
   hPts = scatter(ax, transformedPoints(1,:), transformedPoints(2,:), 50, 'r', 'filled');
else
   hPts = [];
end

% Slider for transparency
uicontrol('Style','text','String','Transparency (Overlay vs Base)', 'Units','normalized','Position',[0.3 0.1 0.4 0.05]);
uicontrol('Style','slider','Min',0,'Max',1,'Value',0.5,'Units','normalized','Position',[0.2 0.05 0.6 0.04],'Callback',@(src,~) set(hOverlay,'AlphaData',get(src,'Value')));

% Checkbox to toggle points visibility
if ~isempty(hPts)
   uicontrol('Style','checkbox','String','Show Transformed Points','Value',1,'Units','normalized','Position',[0.75 0.05 0.2 0.05],'Callback', @(src,~) togglePoints(src, hPts));
end

hold(ax,'off');

% Callback function for checkbox
function togglePoints(src, scatterHandle)
    if get(src,'Value') == 1
       scatterHandle.Visible = 'on';
    else
       scatterHandle.Visible = 'off';
    end
end
end
