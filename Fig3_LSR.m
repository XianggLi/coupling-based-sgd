% LSR comparison experiments
% vs. with Distance-based and UChi.
% y-axis is function loss
clear all;

%% Define number of replications
n_rep = 10;

% Set up array to store results
n = 1 * 1e6; % dataset size
K = n;
gammavec_len = 5;
k_when_half_all = zeros(gammavec_len, n_rep);
pr_error_all = zeros(K, gammavec_len, n_rep);
iter_error_all = zeros(K, gammavec_len, n_rep);

%% Loop through each replication
for i_rep = 1:n_rep
    % Set random seed for this replication
    rng(i_rep);
    
    pr_error = zeros(K, gammavec_len);
    iter_error = zeros(K, gammavec_len);
    
    % Setup
    % Define problem parameters
    d = 100;  % dimension of matrix H

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
    r_values = [1/2, 1/4, 1/8]; % decrease factors for step-size
    q_values = [1.5, 1.5];
    distance_thresh_values = [0.6, 0.4, 1]; % threshold values
    
    % Define UChi parameters
    UChi_s = 0;
    tau = 0;
    burnin = 1 * 1e3; % 1e3 for d=5 and 50, 2 * 1e3 for d=100

    % Run
    % initial step sizes [averaged 1 / 2R2...
    % ,proximity,distance0.4,UChi.,min()]
    alpha0 = 1/mu;
    gamma_vec = [gamma, gamma, gamma, gamma, min(alpha0,gamma)];
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
    
    % for distance-based
    for i = 1:gammavec_len
        ini_theta_idx(i) = 1;
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
        for i=2:2
             Dk = norm(theta_vec(:,i)-theta_vec1(:,i));
             if Dk <= proximity_thresh_values*ini_diff(i)
                gamma_vec(i) = gamma_vec(i)*r_values(1);
                k_when_half_all(i, i_rep) = k_when_half_all(i, i_rep)+1;
                
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
    
        % Run distance_based diagnostic
        for i=3:3
            if k == ceil(q_values(1)^(k0+1))
                numerator = log(norm(theta_vec(:,i) - theta_mat(:,ini_theta_idx(i),i))) ...
                    - log(norm(theta_mat(:,ceil(k/q_values(1)),i) - theta_mat(:,ini_theta_idx(i),i)));
                denominator = log(q_values(1));
                slope = numerator/denominator;
                if slope < distance_thresh_values(1)
                    gamma_vec(i) = gamma_vec(i)*r_values(1);
                    k_when_half_all(i, i_rep) = k_when_half_all(i, i_rep)+1;
                    ini_theta_idx(i) = k;
                end
                k0 = k0 + 1;
            end
        end

        % Run UChi. diagnostic
        for i=4:4
            if k > 2
                UChi_s = UChi_s + dot(theta_mat(:, k, i)-theta_mat(:, k-1, i), ...
                    theta_mat(:, k-1, i)-theta_mat(:, k-2, i)) / gamma_vec(i)^2;
            end
            if k > tau + burnin && UChi_s < 0
                tau = k;
                UChi_s = 0;
                gamma_vec(i) = gamma_vec(i)*r_values(1);
                k_when_half_all(i, i_rep) = k_when_half_all(i, i_rep)+1;
            end
        end

        % min(1/mu*k,1/2R2)
        for i=5:5
            gamma_vec(i)=min(1/(mu*k), gamma);
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
    
    % For plotting at subsampled iteration indices
%     nIdx = 1e5; % number of subsampled iterations 
%     Idxs = ceil( 10.^(log10(K)/nIdx*(0:nIdx)) ); % indices of subsampled iterations
%     Idxs = min(unique(Idxs),K);

    theta_pr_mat = cumsum(theta_mat, 2)./repmat(1:K, d, 1, gammavec_len);
    pr_error = squeeze( sqrt(sum((theta_pr_mat - theta_star_mat ).^2, 1)) );
%     pr_error = pr_error(Idxs,:);
        
    % Error of un-averaged iterates
    iter_error = squeeze( sqrt(sum((theta_mat - theta_star_mat ).^2, 1)) );

    % Save the PR errors for this replication
    pr_error_all(:,:,i_rep) = pr_error;

    % Save the iter errors for this replication
    iter_error_all(:,:,i_rep) = iter_error;

end

k_when_half_avg = mean(k_when_half_all, 2); % average update times

% Compute average PR error over all replications
pr_error_avg = mean(pr_error_all, 3);

% Compute average iter error over all replications
iter_error_avg = mean(iter_error_all, 3);

% filename = ['LSR_compare_thetaloss_static_d', num2str(d), '.mat'];
% save(filename);

% loadfile = ['LSR_compare_thetaloss_adaptive_beta03_d', num2str(d), '.mat'];
% iter_error_avg_adaptive = load(loadfile, 'iter_error_avg');
data = iter_error_avg_adaptive.iter_error_avg;
data = data(:,2);

%% plot
f=figure(1);
clf;
fontsize = 22;

red = [0.8500, 0.3250, 0.0980]; % reddish
purple = [0.4940, 0.1840, 0.5560]; % purplish
orange = [0.9, 0.5, 0.2]; % orangeish
blue = [0, 0.4470, 0.7410]; % blueish
green = [0, 0.6, 0]; % greenish
black = [0, 0, 0]; % black
colors = [red; purple; orange; green; blue; black];
set(gca,'colororder',colors);

hold on

for i = 1:1
lines(i) = plot(1:K, pr_error_avg(:,i), 'LineWidth', 2, 'Color', colors(i,:), 'LineStyle', ':');
end

for i = 2:2
lines(i) = plot(1:K, iter_error_avg(:,i), 'LineWidth', 2, 'Color', colors(i,:), 'LineStyle', '-');
end

for i = 3:3
lines(i) = plot(1:K, data, 'LineWidth', 2, 'Color', colors(i,:), 'LineStyle', '-');
end

for i = 4:4
    lines(i) = plot(1:K, iter_error_avg(:,3), 'LineWidth', 2, 'Color', colors(i,:), 'LineStyle', '--');
end

for i = 5:5
    lines(i) = plot(1:K, iter_error_avg(:,4), 'LineWidth', 2, 'Color', colors(i,:), 'LineStyle', '-.');
end

for i = 6:6
    lines(i) = plot(1:K, pr_error_avg(:,5), 'LineWidth', 2, 'Color', colors(i,:), 'LineStyle', ':');
end


lgnlabels = {};

lgnlabels{1} = sprintf('averaged, 1/2R^2');
lgnlabels{2} = sprintf('Coupling-based, static, \\beta = 0.1');
lgnlabels{3} = sprintf('Coupling-based, adaptive, initial \\beta = 0.3');
lgnlabels{4} = sprintf('Distance-based');
lgnlabels{5} = sprintf('ISGD^{1/2}');
lgnlabels{6} = sprintf('min(1 / 2R^2,1/ \\mu k)');

xlabel('Iteration k', 'FontSize', fontsize);
ylabel('||\theta_k - \theta^*||^2', 'FontSize', fontsize);
title(sprintf('Least-squares regression. (d = %d)', d));
set(gca, 'FontSize', fontsize+5, 'FontName', 'Times New Roman');
set(gca,'yscale','log');
set(gca,'xscale','log');
legend(lines, lgnlabels, 'FontSize', fontsize, 'Location', 'southwest');

% Set the plot window size
% set(gcf, 'Position', [100, 100, 800, 600]);
pos = f.Position;
f.Position = [pos(1) pos(2) 850 450];

hold off

% filename = ['LSR_compare_thetaloss_d', num2str(d), '.mat'];
% save(filename);
% exportgraphics(f, ['LSR_compare_thetaloss_d', num2str(d), '.png'], 'Resolution', 600)