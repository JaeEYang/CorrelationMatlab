function loadTEMpts(~,~)
% Load registration points in pixels onto corresponding TEM images
    [file,path] = uigetfile('*.csv','Select TEMRgpts.csv');
    if isequal(file,0), return; end
    data.TEMpts = loadCSV(fullfile(path,file));
    
    if isempty(data.TEMimg)
       errordlg('Please load the TEM image first'); return;
    end
    imshow(data.TEMimg,[],'Parent',ax); hold(ax,'on');
    
    N = size(data.TEMpts,2);
    hPts = gobjects(1,N);
    for k = 1:N
       hPts(k) = drawpoint(ax,'Position',data.TEMpts(1:2,k)','Color','g');
    end
    hold(ax,'off');
    
    msgbox('Drag the points to fine-tune TEM registration, then press OK to continue.','Fine-tune TEM points','modal');
    waitfor(msgbox);
    for k = 1:N
       data.TEMpts(1:2,k) = hPts(k).Position';
    end
    
    set(statusText,'String','TEM registration points loaded and fine-tuned.');
end
