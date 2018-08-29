clear;
load('lggp_exp_data.mat');

%% recalculate smse disregarding the boundary points
[indmax, ~] = size(mu_full);

smse_full_= NaN(indmax, 1);
smse_fitc_= NaN(indmax, 1);
smse_kiss_= NaN(indmax, 1);
smse_lgg3_= NaN(indmax, 1);
smse_lgg5_= NaN(indmax, 1);

indi = 1;
indf = 500;

for i = 1:indmax
    smse_full_(i) = immse(Y_test(indi:indf), mu_full(i, indi:indf)')/SIGMA_D^2;
    smse_fitc_(i) = immse(Y_test(indi:indf), mu_fitc(i, indi:indf)')/SIGMA_D^2;
    smse_kiss_(i) = immse(Y_test(indi:indf), mu_kiss(i, indi:indf)')/SIGMA_D^2;
    smse_lgg3_(i) = immse(Y_test(indi:indf), mu_lgg3(i, indi:indf)')/SIGMA_D^2;
    smse_lgg5_(i) = immse(Y_test(indi:indf), mu_lgg5(i, indi:indf)')/SIGMA_D^2;
end

figure(1);
set(gcf,'Units','centimeters');
% set(gcf,'Toolbar','None');
% set(gcf,'MenuBar','None');
set(gcf,'Position',[1,2,20,10]);
set(gcf, 'PaperSize', [20,10]);
set(gcf, 'PaperPosition', [1,2,20,10]);

subplot(121);
% subplot('Position',[0.5, 0.5, 0.8, 0.8]);
loglog(N_VEC(2:end), smse_full_(2:end), 'g-^', 'MarkerSize', 10)
hold on, grid on, axis square
loglog(N_VEC(2:end), smse_fitc_(2:end), 'c-s', 'MarkerSize', 10)
loglog(N_VEC(2:end), smse_kiss_(2:end), 'm-d', 'MarkerSize', 10)
loglog(N_VEC(2:end), smse_lgg3_(2:end), 'r-o', 'MarkerSize', 10)
loglog(N_VEC(2:end), smse_lgg5_(2:end), 'b-x', 'MarkerSize', 10)
legend({'Exact', 'FITC', 'KISS', 'LGSWD-3', 'LGSWD-5', 'Location', 'northeast'}, 'FontSize', 12)
xlabel('Number of training points')
ylabel('SMSE')
% ylim([1e-3, 1e2]);
title('Error')
set(gca,'fontsize',12);

subplot(122)
% subplot('Position',[0.5, 0.1, 0.8, 0.8]);
loglog(N_VEC(2:end), elapsed_full(2:end), 'g-^', 'MarkerSize', 10)
hold on, grid on, axis square
loglog(N_VEC(2:end), elapsed_fitc(2:end), 'c-s', 'MarkerSize', 10)
loglog(N_VEC(2:end), elapsed_kiss(2:end), 'm-d', 'MarkerSize', 10)
loglog(N_VEC(2:end), elapsed_lgg3(2:end), 'r-o', 'MarkerSize', 10)
loglog(N_VEC(2:end), elapsed_lgg5(2:end), 'b-x', 'MarkerSize', 10)
% legend('exact', 'FITC', 'fast-grid', 'LG-3b', 'LG-5b', 'Location', 'southeast')
xlabel('Number of training points')
ylabel('Wall clock time (s)')
title('Run time')
set(gca,'fontsize',12);

set(gcf,'Color',[1,1,1]);
export_fig('simu_error_time.pdf')

%%
smse_full_p1= NaN(indmax, 1);
smse_fitc_p1= NaN(indmax, 1);
smse_kiss_p1= NaN(indmax, 1);
smse_lgg3_p1= NaN(indmax, 1);
smse_lgg5_p1= NaN(indmax, 1);

indi = 1;
indf = 100;

for i = 1:indmax
    smse_full_p1(i) = immse(Y_test(indi:indf), mu_full(i, indi:indf)')/SIGMA_D^2;
    smse_fitc_p1(i) = immse(Y_test(indi:indf), mu_fitc(i, indi:indf)')/SIGMA_D^2;
    smse_kiss_p1(i) = immse(Y_test(indi:indf), mu_kiss(i, indi:indf)')/SIGMA_D^2;
    smse_lgg3_p1(i) = immse(Y_test(indi:indf), mu_lgg3(i, indi:indf)')/SIGMA_D^2;
    smse_lgg5_p1(i) = immse(Y_test(indi:indf), mu_lgg5(i, indi:indf)')/SIGMA_D^2;
end

smse_full_p2= NaN(indmax, 1);
smse_fitc_p2= NaN(indmax, 1);
smse_kiss_p2= NaN(indmax, 1);
smse_lgg3_p2= NaN(indmax, 1);
smse_lgg5_p2= NaN(indmax, 1);

indi = 101;
indf = 200;

for i = 1:indmax
    smse_full_p2(i) = immse(Y_test(indi:indf), mu_full(i, indi:indf)')/SIGMA_D^2;
    smse_fitc_p2(i) = immse(Y_test(indi:indf), mu_fitc(i, indi:indf)')/SIGMA_D^2;
    smse_kiss_p2(i) = immse(Y_test(indi:indf), mu_kiss(i, indi:indf)')/SIGMA_D^2;
    smse_lgg3_p2(i) = immse(Y_test(indi:indf), mu_lgg3(i, indi:indf)')/SIGMA_D^2;
    smse_lgg5_p2(i) = immse(Y_test(indi:indf), mu_lgg5(i, indi:indf)')/SIGMA_D^2;
end

figure(2);
set(gcf,'Units','centimeters');
% set(gcf,'Toolbar','None');
% set(gcf,'MenuBar','None');
set(gcf,'Position',[1,2,20,10]);
set(gcf, 'PaperSize', [20,10]);
set(gcf, 'PaperPosition', [1,2,20,10]);

subplot(121);
loglog(N_VEC(2:end), smse_full_p1(2:end), 'g-^', 'MarkerSize', 10)
hold on, grid on, axis square
loglog(N_VEC(2:end), smse_fitc_p1(2:end), 'c-s', 'MarkerSize', 10)
loglog(N_VEC(2:end), smse_kiss_p1(2:end), 'm-d', 'MarkerSize', 10)
loglog(N_VEC(2:end), smse_lgg3_p1(2:end), 'r-o', 'MarkerSize', 10)
loglog(N_VEC(2:end), smse_lgg5_p1(2:end), 'b-x', 'MarkerSize', 10)
legend('Exact', 'FITC', 'KISS', 'LGSWD-3', 'LGSWD-5', 'Location', 'southeast')
xlabel('Number of training points')
ylabel('SMSE')
ylim([1e-3, 1e2]);
title('Error in region x=[0, 0.2)')
set(gca,'fontsize',12);

subplot(122);
loglog(N_VEC(2:end), smse_full_p2(2:end), 'g-^', 'MarkerSize', 10)
hold on, grid on, axis square
loglog(N_VEC(2:end), smse_fitc_p2(2:end), 'c-s', 'MarkerSize', 10)
loglog(N_VEC(2:end), smse_kiss_p2(2:end), 'm-d', 'MarkerSize', 10)
loglog(N_VEC(2:end), smse_lgg3_p2(2:end), 'r-o', 'MarkerSize', 10)
loglog(N_VEC(2:end), smse_lgg5_p2(2:end), 'b-x', 'MarkerSize', 10)
xlabel('Number of training points')
ylabel('SMSE')
ylim([1e-3, 1e2]);
title('Error in region x=(0.2, 0.4)')
set(gca,'fontsize',12);

set(gcf,'Color',[1,1,1]);
export_fig('simu_error_time_p.pdf')

%%
figure(3);
set(gcf,'Units','centimeters');
% set(gcf,'Toolbar','None');
% set(gcf,'MenuBar','None');
set(gcf,'Position',[1,2,20,20]);
set(gcf, 'PaperSize', [20,20]);
set(gcf, 'PaperPosition', [1,2,20,20]);

X_grid = linspace(0, 1, 1e5);
Y_grid = gen_y(X_grid); % WARNING: copied gen_y, changes may not be tracked!!
clf;

% full GP
ind = 5;
subplot(4,1,1)
hold on
plot(X_grid, Y_grid, 'k-')
plot(X_test, mu_full(ind,:), 'g^', 'MarkerSize', 2)
ylabel('y')
title(sprintf('Exact GP with N = %d', N_VEC(ind)))
set(gca,'fontsize',12);

% FITC
ind = 7;
subplot(4,1,2)
hold on
plot(X_grid, Y_grid, 'k-')
plot(X_test, mu_fitc(ind,:), 'cs', 'MarkerSize', 2)
ylabel('y')
title(sprintf('FITC-GP with N = %d', N_VEC(ind)))
set(gca,'fontsize',12);

% KISS
ind = 7;
subplot(4,1,3)
hold on
plot(X_grid, Y_grid, 'k-')
plot(X_test, mu_kiss(ind,:), 'md', 'MarkerSize', 2)
ylabel('y')
title(sprintf('KISS-GP with N = %d', N_VEC(ind)))
set(gca,'fontsize',12);

% LG-GP
ind = 11;
subplot(4,1,4)
hold on
plot(X_grid, Y_grid, 'k-')
plot(X_test, mu_lgg3(ind,:), 'ro', 'MarkerSize', 2)
plot(X_test, mu_lgg5(ind,:), 'bx', 'MarkerSize', 2)
ylabel('y')
xlabel('x')
title(sprintf('LGSWD-GP with N = %d', N_VEC(ind)))
set(gca,'fontsize',12);

set(gcf,'Color',[1,1,1]);
export_fig('simu_fit.pdf')


%%
function [y] = gen_y(x)
    y = sin(5*pi./(x + 0.1));    
end