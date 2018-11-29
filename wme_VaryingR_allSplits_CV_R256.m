% This script generates text embedding for a p.d. text kernel constructed
% from data-dependent random features map using alignment-aware distance 
% for measuring the similairity between two sentences/documents. 
% Expts B: investigate performance changes when varying R using the 
% parameters learned from 10-folds cross validation with R = 256.
%
% Author: Lingfei Wu
% Date: 11/28/2018

clear,clc
nthreads = 4; % set as many as your cpu cores
parpool('local', nthreads);

addpath(genpath('utilities'));
file_dir = './data_proc';
filename_list = {'twitter'};

randdoc_scheme = 1;     % if 1, RF features - uniform distribution
wordemb_scheme = 1;     % if 1, use pre-trained word2vec
                        % if 2, use pre-trained gloVe
                        % if 3, use pre-trained psl
wordweight_scheme = 1;  % if 1, use nbow
docemb_scheme = 2;      % if 1, use dist directly; 
                        % if 2, use soft-min of dist
                       
DMin = 1;    
R_list = [4 8 16 32 64 128 256 512];
% R_list = [4 8 16 32 64 128 256 512 1024 2048 4096 8192];
for jjj = 1:length(filename_list)
    dataSplit_list = [1 2 3 4 5]; % total 5 different splits for Train/Test
    info = [];
    filename = filename_list{jjj};
    disp(filename);
    if strcmp(filename, 'twitter')
        if wordemb_scheme == 1
            if wordweight_scheme == 1
                if docemb_scheme == 1
                    DMax = 9;
                    gamma = -1;
                    lambda_inverse = 1000;
                elseif docemb_scheme == 2
                    DMax = 9;
                    gamma = 0.1;
                    lambda_inverse = 300000;
                end  
            end
        end
        filename_postfix = '-emd_tr_te_split.mat';
    end  
    
    
    % load data and generate corresponding train and test data
    Data = load(strcat(file_dir,'/',filename,filename_postfix));
    [val_min,val_max,d,nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc] = ...
        wme_GenFea_preproc(Data);
    
    Accu_best_list = zeros(2*length(dataSplit_list),length(R_list));
    telapsed_liblinear_list = zeros(1*length(dataSplit_list),length(R_list));
    real_total_emd_time_list = zeros(1*length(dataSplit_list),length(R_list));
    real_user_emd_time_list = zeros(1*length(dataSplit_list),length(R_list));
    for jj = 1:length(dataSplit_list)
        dataSplit = dataSplit_list(jj);
        for j = 1:length(R_list)
            R = R_list(j)
            
            [trainData, testData, telapsed_fea_gen] = wme_GenFea(Data,...
                gamma,R,DMin,DMax,dataSplit,...
                val_min,val_max,d,nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc,...
                randdoc_scheme,wordweight_scheme);

            disp('------------------------------------------------------');
            disp('LIBLinear on WME by varying number of random documents');
            disp('------------------------------------------------------');
            trainFeaX = trainData(:,2:end);
            trainy = trainData(:,1);
            testFeaX = testData(:,2:end);
            testy = testData(:,1);

            % Linear Kernel
            timer_start = tic;
            s2 = num2str(lambda_inverse);
            s1 = '-s 2 -e 0.0001 -q -c '; % liblinear
%             s1 = '-s 2 -e 0.0001 -n 8 -q -c '; % liblinear with omp
            s = [s1 s2];
            model_linear = train(trainy, sparse(trainFeaX), s);
            [train_predict_label, train_accuracy, train_dec_values] = ...
                    predict(trainy, sparse(trainFeaX), model_linear);
            [test_predict_label, test_accuracy, test_dec_values] = ...
                predict(testy, sparse(testFeaX), model_linear);
            telapsed_liblinear = toc(timer_start);
            Accu_best_list(2*(jj-1)+1,j) = train_accuracy(1);
            Accu_best_list(2*(jj-1)+2,j) = test_accuracy(1);
            telapsed_liblinear_list(jj,j) = telapsed_liblinear;
            real_total_emd_time_list(jj,j) = telapsed_fea_gen.real_total_emd_time;
            real_user_emd_time_list(jj,j) = telapsed_fea_gen.user_emd_time/nthreads;
        end
    end
    info.Accu_best_train_ave = mean(Accu_best_list(1:2:end,:),1);
    info.Accu_best_train_std = std(Accu_best_list(1:2:end,:),1);
    info.Accu_best_test_ave = mean(Accu_best_list(2:2:end,:),1);
    info.Accu_best_test_std = std(Accu_best_list(2:2:end,:),1);
    info.Accu_best_list = Accu_best_list;
    info.real_total_emd_time_list = real_total_emd_time_list;
    info.real_user_emd_time_list = real_user_emd_time_list;
    info.telapsed_liblinear = telapsed_liblinear_list;
    info.R = R_list;
    info.DMin = DMin;
    info.DMax = DMax;
    info.gamma = gamma;
    info.lambda_inverse = lambda_inverse;
    info.randdoc_scheme = randdoc_scheme;
    info.wordemb_scheme = wordemb_scheme;
    info.wordweight_scheme = wordweight_scheme;
    info.docemb_scheme = docemb_scheme;
    disp(info);
    savefilename = [filename '_rd' num2str(randdoc_scheme) ...
        '_we' num2str(wordemb_scheme) '_ww' num2str(wordweight_scheme)...
        '_de' num2str(docemb_scheme) '_VaryingR_allSplits_CV_R256'];
    save(savefilename,'info')

end
delete(gcp);
