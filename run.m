close all
clear 
clc

format long;
format compact;

problems       = {'zdt1','zdt2','zdt3'};
problem_length = length(problems);
dimension      = 10;
popsize        = 20;
totalrun       = 3;

for i = 1 : problem_length
    problem = problems{i};
    fprintf('Running on %s...\n', problem);
        for j = 1 : totalrun
            sop               = testmop(problem, dimension);
            sop.popsize       = popsize;
            [pop, objs]       = temomps(sop);
            popStruct = getPop(size(pop, 1));
            popStruct = assignV(pop, objs, popStruct);
            [~, F] = nonDominatedSort(popStruct);
            x = pop(F{1},:);
            y = objs(F{1},:);
        end
end