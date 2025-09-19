function loadFLMpts(~,~)
% Load registration points in pixels onto corresponding FLM images
    [file,path] = uigetfile('*.csv','Select FLMRgpts.csv');
    if isequal(file,0), return; end
    data.FLMpts = loadCSV(fullfile(path,file));
    
    % Display points on FLM image
    if isempty(data.FLMimg)
       errordlg('Please load the FLM image first'); return;
    end
    imshow(data.FLMimg,[],'Parent',ax); hold(ax,'on');
    
    N = size(data.FLMpts,2);
    hPts = gobjects(1,N); % handles for draggable points
    for k = 1:N
       hPts(k) = drawpoint(ax,'Position',data.FLMpts(1:2,k)','Color','r');
    end
    hold(ax,'off');
    
    % Wait for user to adjust points, then update data.FLMpts
    msgbox('Drag the points to fine-tune FLM registration, then press OK to continue.','Fine-tune FLM points','modal');
    waitfor(msgbox);  % Wait until user closes the message box
    for k = 1:N
       data.FLMpts(1:2,k) = hPts(k).Position';
    end
    
    set(statusText,'String','FLM registration points loaded and fine-tuned.');
end
