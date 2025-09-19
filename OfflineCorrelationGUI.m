function OfflineCorrelationGUI()
% OfflineCorrelationGUI: Interactive FLM-TEM correlation GUI, version _v4
% External functions include loadCSV, loadImage, computeTransform, warpImage, transformPoints, interactiveOverlay
% Author: Jae Yang. The main script and functional scripts were developed
% in MATLAB with additional help from ChatGPT (OpenAI) for code refactoring
% and GUI cleanup

% Define default data folder
    if isdeployed
    % Running as a compiled app → use packaged data
    dataPath = fullfile(ctfroot, 'data');
    else
    % Running inside MATLAB → use local project folder
    dataPath = fullfile(pwd, 'data');
    end

    % Initialize data structure
    data = struct('FLMimg',[],'FLMpts',[],'TEMimg',[],'TEMpts',[],...
                  'FLMdata',[],'M',[],'warpedFLM',[],'transformedPts',[],...
                  'FLMfig',[],'TEMfig',[],'FLMhandles',[],'TEMhandles',[],...
                  'overlayFig',[]);

    % Main GUI
    f = figure('Name','Offline Correlation Tool','NumberTitle','off','Position',[100 100 800 500]);

    % Label text
    statusText = uicontrol('Style','text','Units','normalized',...
        'Position',[0.05 0.05 0.9 0.08],'FontSize',14,'HorizontalAlignment','left');

    % Transformation matrix display
    transformText = uicontrol('Style','edit','Units','normalized',...
        'Position',[0.05 0.15 0.9 0.2],'FontSize',14,...
        'Max',2,'Min',0,'HorizontalAlignment','left','Enable','inactive',...
        'String','Transformation Matrix will appear here.');

    % Buttons
    uicontrol('Style','pushbutton','String','1-Load FLM Image & Points','Units','normalized',...
        'Position',[0.05 0.85 0.4 0.07],'FontSize',16, 'FontWeight', 'bold', 'Callback',@loadFLM);
    uicontrol('Style','pushbutton','String','2-Load TEM Image & Points','Units','normalized',...
        'Position',[0.55 0.85 0.4 0.07],'FontSize',16, 'FontWeight', 'bold','Callback',@loadTEM);
    uicontrol('Style','pushbutton','String','3-Load Optional FLM Points','Units','normalized',...
        'Position',[0.05 0.75 0.4 0.07],'FontSize',16, 'FontWeight', 'bold','Callback',@loadFLMdata);
    uicontrol('Style','pushbutton','String','4-Compute Transform & Warp','Units','normalized',...
        'Position',[0.55 0.75 0.4 0.07],'FontSize',16, 'FontWeight', 'bold','Callback',@computeTransformWarp);
    uicontrol('Style','pushbutton','String','5-Show Overlay','Units','normalized',...
        'Position',[0.05 0.65 0.4 0.07],'FontSize',16, 'FontWeight', 'bold','Callback',@showOverlay);
    uicontrol('Style','pushbutton','String','6-Save Results','Units','normalized',...
        'Position',[0.55 0.65 0.4 0.07],'FontSize',16, 'FontWeight', 'bold','Callback',@saveResults);

    %% Callback Functions

    % Load FLM image & points
    function loadFLM(~,~)
        defaultImageFLM = fullfile(dataPath, 'X7Y6_FLM_RGB_2.tif');
        % Image
        [file,path] = uigetfile({'*.tif;*.png;*.jpg','Image Files'},'Select FLM Image', defaultImageFLM);
        if isequal(file,0), return; end
        data.FLMimg = loadImage(fullfile(path,file));

        % Coordinates
        defaultPtsFLM = fullfile(dataPath, 'Item2_X7Y6_FLM_RegSpread9.csv');
        [fileP,pathP] = uigetfile('*.csv','Select FLMRgpts.csv', defaultPtsFLM);
        if isequal(fileP,0), return; end
        data.FLMpts = loadCSV(fullfile(pathP,fileP));

        % Display in independent figure with interative points
        data.FLMfig = figure('Name','FLM Image','NumberTitle','off');
        imshow(data.FLMimg,[]);
        title('FLM Image - Drag points to fine-tune');
        hold on;
        N = size(data.FLMpts,2);
        data.FLMhandles = gobjects(1,N);
        for k = 1:N
           data.FLMhandles(k) = drawpoint('Position',data.FLMpts(1:2,k)','Color','r');
        end
        hold off;
        msgbox('Drag points to fine-tune FLM registration. Close this window when done.','FLM Fine-Tune','modal');
        set(statusText,'String','FLM image and points loaded. Fine-tune in separate window.');
    end

    % Load TEM image & points 
    function loadTEM(~,~)
        defaultImageTEM = fullfile(dataPath, 'TEM_square6_470x.tif');
        % Image
        [file,path] = uigetfile({'*.tif;*.png;*.jpg','Image Files'},'Select TEM Image', defaultImageTEM);
        if isequal(file,0), return; end
        data.TEMimg = loadImage(fullfile(path,file));

        % Coordinates
        defaultPtsTEM = fullfile(dataPath, 'Item1_ER80_G3_470x_Pt6_TEM_RegSpread9.csv');
        [fileP,pathP] = uigetfile('*.csv','Select TEMRgpts.csv', defaultPtsTEM);
        if isequal(fileP,0), return; end
        data.TEMpts = loadCSV(fullfile(pathP,fileP));

        % Display in independent figure with interative points
        data.TEMfig = figure('Name','TEM Image','NumberTitle','off');
        imshow(data.TEMimg,[]);
        title('TEM Image - Drag points to fine-tune');
        hold on;
        N = size(data.TEMpts,2);
        data.TEMhandles = gobjects(1,N);
        for k = 1:N
           data.TEMhandles(k) = drawpoint('Position',data.TEMpts(1:2,k)','Color','g');
        end
        hold off;
        msgbox('Drag points to fine-tune TEM registration. Close this window when done.','TEM Fine-Tune','modal');
        set(statusText,'String','TEM image and points loaded. Fine-tune in separate window.');
    end

    % Load optional FLM points 
    function loadFLMdata(~,~)
        defaultTPts = fullfile(dataPath, 'FLMImagePts.csv');
        [file,path] = uigetfile('*.csv','Select FLM Points (optional)', defaultTPts);
        if isequal(file,0), return; end
        data.FLMdata = loadCSV(fullfile(path,file));
        set(statusText,'String','Optional FLM points loaded.');
    end

    % Compute transformation 
    function computeTransformWarp(~,~)
        % Update points after fine-tuning
        if ~isempty(data.FLMhandles)
            for k = 1:numel(data.FLMhandles)
               data.FLMpts(1:2,k) = data.FLMhandles(k).Position';
            end
        end
        if ~isempty(data.TEMhandles)
            for k = 1:numel(data.TEMhandles)
               data.TEMpts(1:2,k) = data.TEMhandles(k).Position';
            end
        end

        if isempty(data.FLMpts) || isempty(data.TEMpts)
            errordlg('Please load and fine-tune FLM and TEM points first.'); return;
        end
        data.M = computeTransform(data.FLMpts,data.TEMpts);

        % Display the transformation matrix
        matStr = sprintf('%.4f\t%.4f\t%.4f\n%.4f\t%.4f\t%.4f\n%.4f\t%.4f\t%.4f', data.M');
        set(transformText,'String',matStr);

        if isempty(data.FLMimg) || isempty(data.TEMimg)
           errordlg('Please load both images first.'); return;
        end
        data.warpedFLM = warpImage(data.FLMimg,data.M,size(data.TEMimg));

        % Transform optional points
        if ~isempty(data.FLMdata)
           data.transformedPts = transformPoints(data.M,data.FLMdata);
        else
           data.transformedPts = [];
        end
        set(statusText,'String','Transformation computed, FLM warped, and optional points transformed.');
    end

    % Show overlay 
    function showOverlay(~,~)
        if isempty(data.warpedFLM)
           errordlg('Please compute transformation first.'); return;
        end
        data.overlayFig = interactiveOverlay(data.TEMimg,data.warpedFLM,data.transformedPts);
        set(statusText,'String','Interactive overlay opened.');
    end

    % Save results 
    function saveResults(~,~)
        outputDir = uigetdir(pwd,'Select folder to save results');
        if outputDir==0, return; end
        writematrix(data.FLMpts', fullfile(outputDir,'FLMRgpts.csv'));
        writematrix(data.TEMpts', fullfile(outputDir,'TEMRgpts.csv'));
        if ~isempty(data.FLMdata)
            writematrix(data.transformedPts', fullfile(outputDir,'TransformedData.csv'));
        end
        imwrite(data.warpedFLM, fullfile(outputDir,'OverlayFLMTEM.tif'));
        set(statusText,'String',['Results saved to: ', outputDir]);
    end

end
