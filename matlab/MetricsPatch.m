%close all
clc

%%%% SELECT
EPOCHS = 500;
MODEL = 'E500BIGAN';
dataset = 'dataset3';
metric = 2;

% DO NOT CHANGE
mode = 'Test';

n_thresholds = 10000;
path = sprintf('./../E%dResult', EPOCHS );


name = sprintf('%s/%s_novel_%s_%s.txt',path,MODEL,mode,dataset);
novel = load(name);
name = sprintf('%s/%s_normal_%s_%s.txt',path,MODEL,mode,dataset);
normal = load(name);


normal = normal(:,metric);
novel = novel(:,metric);

[p,n,tp,tn,fp,fn,acc, precision, sensitivity, specificity,fscore,mcc,threshold] = ComputeMetricsPatch( normal, novel, n_thresholds );
 
%%Horizontal: fp, vertical tp 2018Wang_NoveltyDetection, 2019Abati
hold on
plot(fp/n,tp/p,'LineWidth',3,'color','r')
grid on
xlabel('True positive rate ')
ylabel('False positive rate ')
set(gca,'FontSize',18)
auc = abs(trapz(fp/n,tp/p))
plot([0 1],[0 1],'color',[0.5 0.5 0.5])
cmd  = sprintf('izi_f(AUC=%0.4f)',auc)
legend(cmd)





