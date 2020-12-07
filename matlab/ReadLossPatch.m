clear all
clc
close all

%%%% SELECT
MODEL = 'E500BIGAN';
mode = 'Test';
dataset = 'dataset3'
bin = 50;
metric = 1; %1,2

path = './../E500Result';
name = sprintf('%s/%s_novel_%s_%s.txt',path,MODEL,mode,dataset);
novel = load(name);
name = sprintf('%s/%s_normal_%s_%s.txt',path,MODEL,mode,dataset);
normal = load(name);

histogram(normal(:,metric),bin)
hold on
histogram(novel(:,metric),bin)

%title(mode)
xlabel('Anomaly Score')
ylabel('Patches')
grid on
legend('Normal','Anomaly')
set(gca,'FontSize',18)
name = sprintf('%s/%s_hist_%s_patch.png', path,MODEL,mode );
saveas(gcf,name)
