clear all
clc
close all

%%%% SELECT
MODEL = 'E500BIGAN' %'E10000BIGAN';
mode = 'Test';
dataset = 'dataset3'
bin = 100;
metric = 1; %1,2

path = './../Result';
name = sprintf('%s/%s_novel_%s_%s.txt',path,MODEL,mode,dataset);
novel = load(name);
name = sprintf('%s/%s_normal_%s_%s.txt',path,MODEL,mode,dataset);
normal = load(name);

histogram(normal(:,metric),bin,'Normalization','probability')
hold on
histogram(novel(:,metric),bin,'Normalization','probability')

%title(mode)
xlabel('Anomaly Score')
ylabel('Patches')
grid on
legend('Normal','Anomaly')
set(gca,'FontSize',18)
name = sprintf('%s/%s_hist_%s_patch.png', path,MODEL,mode );
saveas(gcf,name)
