% This script generates text embedding for a p.d. text kernel constructed
% from data-independent random features map using alignment-aware distance 
% for measuring the similairity between two sentences/documents. 
% We use Liblinear to perform grid search with K-fold cross-validation!
%
% Author: Lingfei Wu
% Date: 11/28/2018

clear,clc
parpool('local');

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
                       
if docemb_scheme == 2
    gamma_list = [1e-2 3e-2 7e-2 0.10 0.19 0.28 0.39 0.56 0.79 1.12 1.58];
elseif docemb_scheme == 1
    gamma_list = -1;
end
DMin = 1;
DMax_list = [3 6 9 12 15 18 21];
R = 256; % number of random documents generated
dataSplit = 1; % we have total 5 different data splits for Train/Test
CV = 10; % number of folders of cross validation
for jjj = 1:length(filename_list)
    filename = filename_list{jjj};
    disp(filename);
    if strcmp(filename, 'twitter')
        filename_postfix = '-emd_tr_te_split.mat';
    end   
    
    % load the train data
    timer_start = tic;
    Data = load(strcat(file_dir,'/',filename,filename_postfix));
    TR_index = Data.TR;
    if size(TR_index,1) == 1
        dataSplit = 1;
    end
    train_words = Data.words(TR_index(dataSplit,:));
    train_BOW_X = Data.BOW_X(TR_index(dataSplit,:));
    train_X = Data.X(TR_index(dataSplit,:));
    train_Y = Data.Y(TR_index(dataSplit,:));
    telapsed_data_load = toc(timer_start)

    [val_min,val_max,d,nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc] = ...
        wme_GenFea_preproc(Data);
    train_NBOW_X = nbow_X_allDoc(Data.TR(dataSplit,:));
    train_IDF_X = idf_X_allDoc(Data.TR(dataSplit,:));
    train_TFIDF_X = tf_idf_X_allDoc(Data.TR(dataSplit,:));
    
    info.aveAccu_best = 0;
    info.valAccuHist = [];
    info.DMaxHist = [];
    info.lambda_invHist = [];
    for jj = 1:length(DMax_list)
    for j = 1:length(gamma_list)
        DMax = DMax_list(jj)
        gamma = gamma_list(j)
       
        % shuffle the train data
        shuffle_index = randperm(length(train_Y)); 
        X = train_X(shuffle_index);
        Y = train_Y(shuffle_index);
        NBOW_X = train_NBOW_X(shuffle_index);
        IDF_X = train_IDF_X(shuffle_index);
        TFIDF_X = train_TFIDF_X(shuffle_index);
        N = size(X,2);
        trainData = zeros(N,R+1);
        rng('default')
        timer_start = tic;
        if randdoc_scheme == 1 
            % Method 1: RF features - uniform distribution. Generate random 
            % features based on emd distance between original documents and 
            % random documents where random words are sampled in R^d word space
            sample_X = cell(1,R);
            sample_weight_X = cell(1,R);
            for i=1:R
                D = randi([DMin,DMax],1);
    %             sample_X{i} = randn(d,D)./sigma; % gaussian
                sample_X{i} = val_min+(val_max-val_min)*(rand(d,D)); % 
                % uniform normalize random word vector into an unit vector 
                % to be consistent with pre-trained words in word2vector space
                for ii=1:D
                    sample_X{i}(:,ii) = sample_X{i}(:,ii)/norm(sample_X{i}(:,ii));
                end
                sample_weight_X{i} = ones(1,D); % uniform frequence for random word
            end
        end
        
        if wordweight_scheme == 1 % use NBOW
            weight_X = NBOW_X;
        elseif wordweight_scheme == 2 % use TFIDF
            weight_X = TFIDF_X;
        end
        trainFeaX_random = wmd_dist(X,weight_X,sample_X,sample_weight_X,gamma);
        trainFeaX_random = trainFeaX_random/sqrt(R); 
        trainData(:,2:end) = trainFeaX_random;
        trainData(:,1) = Y;
        telapsed_fea_gen = toc(timer_start);

        disp('------------------------------------------------------');
        disp('LIBLinear performs basic grid search by varying lambda');
        disp('------------------------------------------------------');
        % Linear Kernel
        lambda_inverse = [1e2 3e2 5e2 8e2 1e3 3e3 5e3 8e3 1e4 3e4 5e4 8e4...
              1e5 3e5 5e5 8e5 1e6 1e7];
        for i=1:length(lambda_inverse)
            valAccu = zeros(1, CV);
            for cv = 1:CV
                subgroup_start = (cv-1) * floor(N/CV);
                mod_remain = mod(N, CV);
                div_remain = floor(N/CV);
                if  mod_remain >= cv
                    subgroup_start = subgroup_start + cv;
                    subgroup_end = subgroup_start + div_remain;
                else
                    subgroup_start = subgroup_start + mod_remain + 1;
                    subgroup_end = subgroup_start + div_remain -1;
                end
                test_indRange = subgroup_start:subgroup_end;
                train_indRange = setdiff(1:N,test_indRange);
                trainFeaX = trainData(train_indRange,2:end);
                trainy = trainData(train_indRange,1);
                testFeaX = trainData(test_indRange,2:end);
                testy = trainData(test_indRange,1);
                
                s2 = num2str(lambda_inverse(i));
                s1 = '-s 2 -e 0.0001 -q -c '; % liblinear
%                 s1 = '-s 2 -e 0.0001 -n 8 -q -c '; % liblinear with omp
                s = [s1 s2];
                timer_start = tic;
                model_linear = train(trainy, sparse(trainFeaX), s);
                [test_predict_label, test_accuracy, test_dec_values] = ...
                    predict(testy, sparse(testFeaX), model_linear);
                telapsed_liblinear = toc(timer_start);
                valAccu(cv) = test_accuracy(1);             
            end
            ave_valAccu = mean(valAccu);
            std_valAccu = std(valAccu);
            if(info.aveAccu_best+0.1 < ave_valAccu)
                info.DMaxHist = [info.DMaxHist;DMax];
                info.lambda_invHist = [info.lambda_invHist;lambda_inverse(i)];
                info.valAccuHist = [info.valAccuHist;valAccu];
                info.valAccu = valAccu;
                info.aveAccu_best = ave_valAccu;
                info.stdAccu = std_valAccu;
                info.telapsed_fea_gen = telapsed_fea_gen;
                info.telapsed_liblinear = telapsed_liblinear;
                info.runtime = telapsed_fea_gen + telapsed_liblinear;
                info.gamma = gamma;
                info.R = R;
                info.DMin = DMin;
                info.DMax = DMax;
                info.lambda_inverse = lambda_inverse(i);
                info.randdoc_scheme = randdoc_scheme;
                info.wordemb_scheme = wordemb_scheme;
                info.wordweight_scheme = wordweight_scheme;
                info.docemb_scheme = docemb_scheme;
            end
        end
    end
    end
    disp(info);
    savefilename = [filename '_rd' num2str(randdoc_scheme) ...
        '_we' num2str(wordemb_scheme) '_ww' num2str(wordweight_scheme)...
        '_de' num2str(docemb_scheme) '_R' num2str(R) '_'  num2str(CV) 'fold_CV'];
    save(savefilename,'info');
end
delete(gcp);
