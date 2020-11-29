%close all
clc

%%%% SELECT
EPOCHS = 500;
MODEL = 'IZIf';


% DO NOT CHANGE
mode = 'Test';

n_thresholds = 256;
path = sprintf('./../E%d_Results', EPOCHS );

oname = sprintf('%s/%s_metrics_%s_patch.txt',path,MODEL,mode);
fileID = fopen( oname, 'w' );

fprintf(fileID,'\n%s\n',mode);
name = sprintf('%s/%s_loss_novel_%s.txt',path,MODEL,mode);
novel = load(name);
name = sprintf('%s/%s_loss_normal_%s.txt',path,MODEL,mode);
normal = load(name);

% RULE 
%novel = sum(novel,2);
%normal = sum(normal,2);


[p,n,tp,tn,fp,fn,acc, precision, sensitivity, specificity,fscore,mcc,threshold] = ComputeMetricsPatch( normal, novel, n_thresholds );
 

hold on
plot(fn/p,tn/n,'LineWidth',3,'color','r')
grid on
xlabel('False Negative Rate ')
ylabel('True Negative Rate ')
set(gca,'FontSize',18)
auc = abs(trapz(fn/p,tn/n))
plot([0 1],[0 1],'color',[0.5 0.5 0.5])
cmd  = sprintf('izi_f(AUC=%0.4f)',auc)
legend(cmd)

name = sprintf('%s/%s_auc_%s_patch.png', path,MODEL,mode );
saveas(gcf,name)


fprintf(fileID,'\nMaximum acc\n');
%[max_acc,id]=max(acc);
% Provitional solution
index = find( acc == max(acc) );
if length(index) == 1
    id = index;
else
    id = floor((index(end)-index(1))/2);
end
fprintf(fileID,'max_acc     = %f\n', acc(id) );
fprintf(fileID,'Precison    = %f\n', precision(id) );
fprintf(fileID,'Sensitivity = %f\n', sensitivity(id) );
fprintf(fileID,'Specificity = %f\n', specificity(id) );
fprintf(fileID,'Fscore      = %f\n', fscore(id) );
fprintf(fileID,'MCC         = %f\n', mcc(id) );
fprintf(fileID,'Threshold   = %f\n', threshold(id));
  
fclose(fileID);
cmd = sprintf('%s is ready!!!',oname);
disp(cmd)

