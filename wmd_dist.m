% This script generates text embedding for a p.d. text kernel constructed
% from data-dependent random features map using alignment-aware distance 
% for measuring the similairity between two sentences/documents. 
%
% Author: Lingfei Wu
% Date: 11/28/2018

function [KMat, user_emd_runtime] = wmd_dist(newX, weight_newX, baseX, ...
    weight_baseX, gamma)
    
    [~, N1] = size(newX);
    [~, N2] = size(baseX);
    
    KMat = zeros(N1,N2);
    user_emd_runtime = 0;
    tic;
    parfor i = 1:N1
        Ei = zeros(1,N2);
%         x1 = weight_newX{i};
        x1 = weight_newX{i}./sum(weight_newX{i});
        data1 = newX{i};
        for j = 1:N2
            if isempty(weight_newX{i}) || isempty(weight_baseX{j})
                Ei(j) = Inf; 
            else
%                 x2 = weight_baseX{j};
                x2 = weight_baseX{j}./sum(weight_baseX{j});
                data2 = baseX{j};
                D = distance(data1,data2);
                D(D < 0) = 0;
                D = sqrt(D);
                emd_telapsed = tic;
                emd = emd_mex(x1,x2,D); % use classic emd from Yossi Rubner
                user_emd_runtime = user_emd_runtime + toc(emd_telapsed);                
                if gamma == -1
                    Ei(j) = emd; % directly use emd as features
                else
                    Ei(j) = exp(-emd*gamma); % use soft-min features of emd
                end
            end
        end
        KMat(i,:) = Ei;
    end
    toc
    
end
