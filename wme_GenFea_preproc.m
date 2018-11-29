% This script generates statistical information about the data for
% generating random documents using different feature maps
%
% Author: Lingfei Wu
% Date: 11/28/2018

function [val_min,val_max,d,nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc] = ...
    wme_GenFea_preproc(Data)

    % analyze the number of texts and maxmimum number of words in each text
    timer_start = tic;
    N_allDoc = length(Data.Y); % get the number of texts
    d = size(Data.X{1},1); % get the dimension of continuous word vector
    L = 0; % get maxmimum number of words in each text
    val_min = 1e8;
    val_max = -1e8;
    nonEmptyWords = [];
    L_list = zeros(1,N_allDoc);
    for i=1:N_allDoc
        if isempty(Data.BOW_X{i})
            fprintf('Warning: Texts %d is empty!\n', i);
        else
            L_temp = size(Data.X{i},2);
            L_list(i) = L_temp;
            Data.BOW_X{i} = Data.BOW_X{i}(1:L_temp);
            if L < L_temp
                L = L_temp;
            end
            val_max_temp = max(max(Data.X{i}));
            val_min_temp = min(min(Data.X{i}));
            if val_max_temp > val_max
                val_max = val_max_temp;
            end
            if val_min_temp < val_min
                val_min = val_min_temp;
            end
            nonEmptyWords = [nonEmptyWords Data.words{i}(1:size(Data.X{i},2))];
        end
    end
    [uniqueWords,IA,IC] = unique(nonEmptyWords);
    BOW_dim = length(uniqueWords);
    fprintf('Data N=%d, BOW Dim=%d, L=%d, L_ave=%.3d, d=%d, val_min=%f, val_max=%f \n', ...
        N_allDoc, BOW_dim, L, mean(L_list), d, val_min, val_max); 
    telapsed_data_analysis = toc(timer_start)
    
    % Build a map for word and IDF so we can easily retrieve word count 
    timer_start = tic;
    keySet = uniqueWords;
    valueSet = zeros(1,BOW_dim);
    uniqueWords_idf_map = containers.Map(keySet,valueSet);
    for j=1:N_allDoc
        if isempty(Data.BOW_X{j})
%             fprintf('Warning: Texts %d is empty!\n', j);
        else
            for i=1:size(Data.X{j},2)
                uniqueWords_idf_map(Data.words{j}{i}) = ...
                    uniqueWords_idf_map(Data.words{j}{i}) + 1;
            end
        end
    end

    % generate NBOW and TF-IDF model weights for train and test data
    nbow_X_allDoc = cell(1,N_allDoc);
    idf_X_allDoc = cell(1,N_allDoc);
    tf_idf_X_allDoc = cell(1,N_allDoc);
    for j=1:N_allDoc
        keys = Data.words{j}(1:size(Data.X{j},2));
        df = cell2mat(values(uniqueWords_idf_map,keys));
        idf_X_allDoc{j} = log(1+N_allDoc./df);
        nbow_X_allDoc{j} = Data.BOW_X{j}./sum(Data.BOW_X{j});
        tf_idf_X_allDoc{j} = nbow_X_allDoc{j}.*idf_X_allDoc{j};
    end
    telapsed_data_analysis_tfidf = toc(timer_start)
end