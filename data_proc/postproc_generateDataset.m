% function [ret] = postproc_20ng(newfilename,oldfilename,file_dir,split)
% 20NG_POSTPROC limits all 20news documents to the most common 500 words in
% each document
%     ret = 0; % return success or not
%
% Author: Lingfei Wu
% Date: 11/28/2017

    clear,clc
    split_ratio = 0.7; % train/test split ratio
    file_dir = './'; 
    newfilename = 'twitter-emd_tr_te_split.mat';
    oldfilename = 'twitter.mat';
    olddata = load(strcat(file_dir,'/',oldfilename));
    
    BOW_X_new = olddata.BOW_X;
    BOW_X = BOW_X_new;
    words_new = olddata.words;
    words = words_new;
    X_new = olddata.X;
    X = X_new;
    Y = olddata.Y;
    C = olddata.C;
    
    % create 5 different train/test splits if no original data split
    num_docs = length(X_new);
    train_num = ceil(num_docs * split_ratio);
    IndexMatrix = [ randperm(num_docs);      
                    randperm(num_docs);
                    randperm(num_docs);
                    randperm(num_docs);
                    randperm(num_docs);];
    TE = IndexMatrix(:,1:train_num);
    TR = IndexMatrix(:,train_num+1:num_docs);
    
    % keep most frequent words in a document in case of long document
    numWordsToKeep = 500;
    for i = 1:num_docs
        [temp,I] = sort(BOW_X_new{i},'descend');
        L = length(I); % number of total words in a document
        if L >= numWordsToKeep
            BOW_X{i} = BOW_X_new{i}(I);
            BOW_X{i} = BOW_X_new{i}(1:numWordsToKeep);
        end
        words{i} = words_new{i}(1:length(I));
        if L >= numWordsToKeep
            words{i} = words{i}(I);
            words{i} = words{i}(1:numWordsToKeep);
        end
        if L >= numWordsToKeep
            X{i} = X_new{i}(:,I);
            X{i} = X_new{i}(:,1:numWordsToKeep);
        end
    end
    
    % save the dataset as the inputs for the main program
    save(newfilename,'BOW_X','words','X','Y','C','TE','TR','-v7.3');

