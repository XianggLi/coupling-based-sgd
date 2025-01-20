% LSR setting: illustrates the exponentially decay of distance ratio
clear all;

%% Define number of replications
n_rep = 10;

% Set up array to store results
n = 1 * 1e6; % dataset size
K = n;
gammavec_len = 6;
k_when_half_all = zeros(gammavec_len, n_rep);
k_when_half_one = [];
Dk_list = [];
pr_error_all = zeros(K, gammavec_len, n_rep);
iter_error_all = zeros(K, gammavec_len, n_rep);

%% Loop through each replication
for i_rep = 1:n_rep
    % Set random seed for this replication
    rng(i_rep);
    
    % Setup
    % Define problem parameters
    d = 5;  % dimension of matrix H

    sigma = 1; % noise level
    
    % Generate data
    eigenvals = 1./linspace(1,d,d); % eigenvalues
    eigenvecs = orth(randn(d,d)); % eigenvectors
    H = eigenvecs * diag(eigenvals) * eigenvecs';
    eig_H = eig(diag(eigenvals));  % compute eigenvalues of H
    R2 = trace(H); % trace of H
    
    theta_star = randn(d,1); % true parameter vector
    
    % generate inputs xi
    x = randn(n, d) * chol(H);
    % generate outputs y following the generative model
    y = x * theta_star + sigma * randn(n, 1);
    
    % Define SGD parameters
    gamma = 1/(2*R2); % initial step sizes
    mu = 1/d; % min(eig_H), this is strong convexity constant
    
    % Define proximity_based parameters
    proximity_thresh_values = 0.1;
    bsteps = 1*1e2;

    % Define distance_based parameters
    k0 = 5;
    r_values = [1/4, 1/2, 1/8]; % decrease factors for step-size
    q_values = [1.5, 1.5];
    distance_thresh_values = [0.6, 0.4, 1]; % threshold values
    
    % Define UChi parameters
    UChi_s = 0;
    tau = 0;
    burnin = 1 * 1e3;

    % Run
    % initial step sizes [averaged 1 / 2R2...
    % ,proximity,distance0.4,UChi.,min()]
    gamma_vec = [gamma gamma gamma/4 gamma/16 gamma/64 gamma/256];
    gammavec_len = length(gamma_vec);
    
    % first theta sequence
    theta_vec = zeros(d, gammavec_len);
    theta_mat = zeros(d, K, gammavec_len); % theta_vec sequence
    
    % second theta sequence
    theta_vec1 = ones(d, gammavec_len);
    theta_mat1 = ones(d, K, gammavec_len);
    for i = 1:gammavec_len
        ini_diff(i) = norm(theta_vec(:,i)-theta_vec1(:,i)); %initial difference
    end

    tic;
    fprintf('Iteration progress: 00.00%%');

    for k = 1:K
        % calculate gradients from mini-batch data
        i = randi(n);
%         i = k;
        grad = x(i,:)'*(x(i,:)*theta_vec - y(i));
        grad1 = x(i,:)'*(x(i,:)*theta_vec1 - y(i));
    
        % update theta_vec
        theta_vec = theta_vec - gamma_vec.*grad;
        theta_mat(:,k,:) = theta_vec;

        % update second theta_vec sequence
        theta_vec1 = theta_vec1 - gamma_vec.*grad1;
        theta_mat1(:,k,:) = theta_vec1;

        % Run proximity_based diagnostic
        for i=1:1
             Dk = norm(theta_vec(:,i)-theta_vec1(:,i));
             if i_rep == 1
                 Dk_list = [Dk_list, Dk/ini_diff(1)];
             end
             if Dk <= proximity_thresh_values*ini_diff(i)
                gamma_vec(i) = gamma_vec(i)*r_values(1);
                k_when_half_all(i, i_rep) = k_when_half_all(i, i_rep)+1;
                if i_rep == 1
                    k_when_half_one = [k_when_half_one, k];
                end
                
                % reset the vec sequence initial value as the other sequence at #bsteps steps backward
                reset_ini = k - bsteps;
                if reset_ini <= 0
                    reset_ini = k;
                end
                theta_vec1(:,i) = theta_mat1(:,reset_ini,i);
                theta_mat1(:,k,i) = theta_vec1(:,i);
                
                %re-compute the initial difference
                ini_diff(i) = norm(theta_vec(:,i)-theta_vec1(:,i));
            end
        end

        % Show progress
        if mod(k, 1e4) == 0
            fprintf("%05.2f%%\n", k/K*100);
        end
        
    end
    fprintf("\n");
    clear rand_nums
    toc;
    
    % Compute their errors
    theta_star_mat = repmat(theta_star, 1, K, gammavec_len);
    
%     % For plotting at subsampled iteration indices
%     nIdx = K; % number of subsampled iterations 
%     Idxs = ceil( 10.^(log10(K)/nIdx*(0:nIdx)) ); % indices of subsampled iterations
% %     Idxs = min(unique(Idxs),K);
%     Idxs = unique(Idxs);

%     theta_pr_mat = cumsum(theta_mat, 2)./repmat(1:K, d, 1, gammavec_len);
%     pr_error = squeeze( sqrt(sum((theta_pr_mat - theta_star_mat ).^2, 1)) );
%     pr_error = pr_error(Idxs,:);
        
    % Error of un-averaged iterates
    iter_error = squeeze( sqrt(sum((theta_mat - theta_star_mat ).^2, 1)) );

    % Save the PR errors for this replication
%     pr_error_all(:,:,i_rep) = pr_error;

    % Save the iter errors for this replication
    iter_error_all(:,:,i_rep) = iter_error;

end

k_when_half_avg = mean(k_when_half_all, 2); % average update times

% Compute average PR error over all replications
pr_error_avg = mean(pr_error_all, 3);

% Compute average iter error over all replications
iter_error_avg = mean(iter_error_all, 3);

% For plotting at subsampled iteration indices
nIdx = 1e3; % number of subsampled iterations 
Idxs = ceil( 10.^(log10(K)/nIdx*(0:nIdx)) ); % indices of subsampled iterations
%     Idxs = min(unique(Idxs),K);
Idxs = unique(Idxs);
iter_error_avg = iter_error_avg(Idxs,:);

%% Plot
tic;

% illustrates the exponentially decay of distance ratio
f=figure(1);
clf;

fontsize = 22;
colors = get(gca,'colororder'); % default line colors
colors = [colors; [0.8, 0.8, 0]]; % append a new 8th color
colors = colors([4,6,3,5,2,1,7,8], :); 

% % Diminishing stepsize
% hdim = plot(Idxs, dimin_error, '-', 'DisplayName',...
%     ['\alpha_k = ', sprintf('%.2f',alpha0), '/sqrt(k)'],...
%         'LineWidth',4,  'MarkerSize', 4, 'color', [.9,.1,.9].^0.3);

hold on;

% Plot Error
for i = 1:1
    % TA
    hta(i) = plot(Idxs, iter_error_avg(:,i),'-',...
        'LineWidth',1, 'color', colors(i,:).^1.1);
    hold on;
end

% for i = 7:8
%     % TA
%     hta(i) = plot(Idxs, iter_error_avg_all(:,i),'-',...
%         'LineWidth',1, 'color', colors(i,:).^1.1);
%     hold on;
% end

for i = 2:gammavec_len
    % TA
    hta(i) = plot(Idxs, iter_error_avg(:,i),'-.',...
        'LineWidth',1, 'color', colors(i,:).^1.1);
    hold on;
end

% Adjust Plot
lgnlabels = {};
% lgnlabels{1} = sprintf('proximity-based, \\gamma = %.5f',gamma_vec(1));
% for i = 2:gammavec_len
%    lgnlabels{i} = sprintf('fixed step-size, \\gamma = %.3f',gamma_vec(i));
% end

% First label
for i = 1:1
    lgnlabels{i} = sprintf('coupling-based, static, initial \\gamma = 1/2R^2, \\beta = 0.1');
end

% for i = 7:8
%     lgnlabels{i} = sprintf('proximity-based, \\beta = %.2f',0.1^(i-6));
% end

% Labels for 1/2R^2, 1/8R^2, 1/32R^2...
for i = 2:gammavec_len
  exponent = 4^(i - 2);
  exponent = exponent*2;
  lgnlabels{i} = sprintf('fixed step-size \\gamma = 1/%dR^2', exponent); 
end

% make duplicate line objects with same colors (for use in legend)
for i = 1:length(hta)
    hcolors(i) = plot([-1,-2],[0,0],'s', 'markersize',20); 
    set(hcolors(i), 'Color', get(hta(i), 'Color'));
    set(hcolors(i), 'MarkerFaceColor', get(hta(i), 'Color'));
    
end

% make duplicate line objects with same line styles (for use legend)
% hstemp = [hta(1), htarr(1), htarr3(1), htarr4(1), htarr5(1), htarr6(1)];
hstemp = [hta(1)];
for i = 1:length(hstemp)
    hlstyles(i) = plot([-1,-2],[0,0],'k'); 
    set(hlstyles(i), 'linestyle', get(hstemp(i), 'linestyle'));
    set(hlstyles(i), 'linewidth', get(hstemp(i), 'linewidth'));
end
% hdk = plot(1:K, Dk_list,':');

for i = 1:length(k_when_half_one)
    xline(k_when_half_one(i)); % Draw a vertical line at each x-value
end

hleg = legend('FontSize', fontsize, 'Location', 'southwest', 'FontName', 'Times New Roman');
% lgnlabels{end+1} = ['\alpha_k = ', sprintf('%d',alpha0), '/\surdk'];
% lgnlabels{end+1} = 'D_k/D_0';
legend(hta, lgnlabels);

xlabel('Iteration k', 'FontSize', fontsize);
ylabel('||\theta_k - \theta^*||^2', 'FontSize', fontsize);
% title(sprintf('Initial ratio threshold = %.2f, Backwards steps = %d', proximity_thresh_values(1), bsteps));
title(sprintf('Least-squares regression. (d = %d)', d));
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
set(gca,'yscale','log');
set(gca,'xscale','log');
set(gca,'xtick',[1e1,1e2,1e3,1e4,1e5,1e6])
xlim([1e1,K]);
% ylim([3e-4, 3e1]);
set(gca,'ytick',[1e-3, 1e-2, 1e-1, 1e0, 1e2])
set(gcf, 'Color', 'white');
% y axis on both sides
add = gca;
extra = axes('Position',get(add,'Position'),'FontSize', get(add,'FontSize'),...
    'FontName', get(add,'FontName'), 'ytick', get(add,'ytick'), 'yscale', get(add,'yscale'), ...
    'Color', 'none','XTick',[],'YAxisLocation','right');
linkaxes([add extra],'xy');

pos = f.Position;
f.Position = [pos(1) pos(2)  800 450];