function [xtrain, ftrain] = temomps(problem)
    global nFEs params ;
    
    % set the algorithm parameters
    [db_x, db_f] = init(problem);
    ftrain = db_f.ftrain;
    xtrain = db_x.xtrain;
    pop = db_x.xtrain;
    
    while nFEs < params.maxFEs
        weight = rand(1, problem.od);
        weight = weight / sum(weight);
        rng('shuffle');
        ytrain = aggregate(ftrain, weight);
        ysrc1  = aggregate(db_f.fsrc1, weight);
        ysrc2  = aggregate(db_f.fsrc2, weight);
        y_min  = min([ytrain; ysrc1; ysrc2]);
        
        % src1 building
        ysrc1 = normalize(ysrc1, []);
        src_models{1} = fitrgp(db_x.xsrc1, ysrc1);
        % src2 building
        ysrc2 = normalize(ysrc2, []);
        src_models{2} = fitrgp(db_x.xsrc2, ysrc2);
        
        [~, ytrain] = normalize([], ytrain);
        model = tsgp(xtrain, ytrain, src_models);
        
        % select pop from training set
        [~, idx] = sort(ytrain);
        idx = idx(1:params.popsize);
        pop = xtrain(idx, :);
        
        % reproduction, selection
        pop1 = de_op(pop, problem);
        pop2 = blx_op(pop, problem);
        pop3 = sbx_op(pop, problem);
        popall = [pop1; pop2; pop3];
        [m, s] = model.predict(popall);
        EI = ei(m, s, y_min);
        [~, idx] = max(EI);
        soi = popall(idx, :);
        
        f_soi = problem.func(soi);
        nFEs = nFEs + 1;
        xtrain = [xtrain; soi];
        ftrain = [ftrain; f_soi];
        disp(['Dimension = ' num2str(problem.pd) ': FE = ' num2str(nFEs)]);
    end
end

%% initialisation process
function [db_x, db_f] = init(problem)
    global nFEs params;
    
    % parameter settings
    params.popsize  = problem.popsize;  % population size of DE
    params.maxFEs   = 600;
    
    ntrain = params.popsize;
    nsrc1  = 250;
    nsrc2  = 250;
        
    % training data
    % Target Task
    xtrain = lhsdesign(ntrain, problem.pd);
    ftrain = problem.func(xtrain);
    nFEs = ntrain;
        
    % src1
    x_src1 = lhsdesign(nsrc1, problem.pd);
    f_src1 = problem.func(x_src1);
    nFEs = nFEs + nsrc1;
    % src2
    x_src2 = lhsdesign(nsrc1, problem.pd);
    f_src2 = problem.func(x_src2);
    nFEs = nFEs + nsrc2;
    
    db_x.xtrain = xtrain;
    db_x.xsrc1  = x_src1;
    db_x.xsrc2  = x_src2;
    db_f.ftrain = ftrain;
    db_f.fsrc1  = f_src1;
    db_f.fsrc2  = f_src2;
end

function y = aggregate(f, w)
    % Tchebycheff Aggregation
    m = size(f, 2);
    for i = 1 : m
        tmp(:, i) = f(:,i) * w(i);
    end
    y = max(tmp, [], 2);
end

function [ytest,ytrain] = normalize(ytest, ytrain)
    y=[ytest; ytrain];
    miny = min(y);
    maxy = max(y);
    y = (y-miny) / (maxy-miny);
    
    ntest = length(ytest);
    ytest = y(1 : ntest);
    ytrain = y(ntest+1 : end);
end

function EI = ei(mean, s, f_min)
    diff = f_min - mean;
    norm = diff ./ s;
    EI = diff .* normcdf(norm) + s .* normpdf(norm);
end
