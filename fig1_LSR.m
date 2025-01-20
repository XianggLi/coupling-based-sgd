% Test coupling idea for detecting stationarity
clear all;

%% Define number of replications
n_rep = 10;

% Set up array to store results
n = 1 * 1e6; % dataset size
K = n;

iter_error_all = zeros(K, 6, n_rep);
iter_error2_all = zeros(K, 6, n_rep);
ratio_seq_all = zeros(K, 6, n_rep);

%% Loop through each replication
for i_rep = 1:n_rep
    %% Setup
    rng(2)
    
    d = 5;  % dimension of matrix H
    k = 1;
    
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
    
    % initial stepsizes
    alpha_vec = [3.20 3.20 0.1 0.80 0.40 0.01];
    avec_len = length(alpha_vec);
    % alpha0 = 0.95; % deminishing stepsize
    
    % For our alg.
    % Two theta sequences
    theta_vec = zeros(d, avec_len); % latest theta_k for all algorithms
    theta_mat = zeros(d, K, avec_len); % theta_k sequence trajectories for all algorithms
    
    theta_vec1 = ones(d, avec_len);
    theta_mat1 = ones(d, K, avec_len);
    
    for i = 1:avec_len
        ini_diff(i) = norm(theta_vec(:,i)-theta_vec1(:,i)); %initial distance between two sequences
    end
    
    tic;
    fprintf('Iteration progress: 00.00%%');
    while k <= K
     
    %     state_next = find(rand() < P_cum(state,:), 1, 'first');
    %     A = As(:,:, state, state_next);
    %     b = r(state) * Phi(state,:)';
    
        i = randi(n);
        grad = x(i,:)'*(x(i,:)*theta_vec - y(i));
        grad1 = x(i,:)'*(x(i,:)*theta_vec1 - y(i));

        % update theta_vec
        theta_vec = theta_vec - alpha_vec.*grad;
        theta_mat(:,k,:) = theta_vec;
    
        % update second theta_vec sequence
        theta_vec1 = theta_vec1 - alpha_vec.*grad1;
        theta_mat1(:,k,:) = theta_vec1;
        
        k = k + 1;
    %     state = state_next;
    
        % Show progress
        if mod(k, 1e4) == 0
            fprintf("%05.2f%%\n", k/K*100);
        end
    end
    fprintf("\n");
    clear rand_nums
    toc;
    
    %% Compute PR, TA and RR and their errors, removing unused variable as we go
    tic;
%     theta_0_mat = repmat(zeros(d,1), 1, K, avec_len);
    theta_star_mat = repmat(theta_star, 1, K, avec_len);
    ini_diff_mat = repmat(ini_diff, K, 1);
    
    % Error of un-averaged iterates 
    iter_error_all(:,:,i_rep) = squeeze( sqrt(sum((theta_mat - theta_star_mat ).^2, 1)) );
    
    % Error of un-averaged iterates of another sequence
    iter_error2_all(:,:,i_rep) = squeeze( sqrt(sum((theta_mat1 - theta_star_mat ).^2, 1)) );
    
    % ====== ====== %
    % 2-norm of Dk: theta_seq1_k - theta_seq2_k
    ratio_seq_all(:,:,i_rep) = squeeze( sqrt(sum((theta_mat - theta_mat1 ).^2, 1)))./ini_diff_mat;
    % seq_gap_scaled = seq_gap_scaled(Idxs,:);
    
    toc;
end

% Compute average iter error over all replications
iter_error = mean(iter_error_all, 3);
iter_error2 = mean(iter_error2_all, 3);
ratio_seq = mean(ratio_seq_all, 3);

% For plotting at subsampled iteration indices
nIdx = 1e3; % number of subsampled iterations 
Idxs = ceil( 10.^(log10(K)/nIdx*(0:nIdx)) ); % indices of subsampled iterations
Idxs = min(unique(Idxs),K);

iter_error = iter_error(Idxs,:);
iter_error2 = iter_error2(Idxs,:);
ratio_seq = ratio_seq(Idxs,:);

% filename = 'fig1_LSR_250103.mat';
% save(filename);

%% Plot
tic;

% illustrates the observation of convergence time
f=figure(1);
clf;

fontsize = 28;
colors = get(gca,'colororder'); % default line colors
colors = colors([6,4,3,5,2,1,7], :);

% Unaveraged error lines
yyaxis right
for i = 3:3
    line(1) = plot(Idxs, iter_error(:,i), '--', 'DisplayName',['\gamma = ', sprintf('%.3f',alpha_vec(i))],...
       'LineWidth',1.5,  'color', colors(i,:).^1.1);
    hold on
    line(2) = plot(Idxs, iter_error2(:,i), ':', 'DisplayName',['\gamma = ', sprintf('%.3f',alpha_vec(i))],...
       'LineWidth',1.5,  'color', colors(i,:).^1.1);
    hold on
end

for i = 6:6
    line(4) = plot(Idxs, iter_error(:,i), '--', 'DisplayName',['\gamma = ', sprintf('%.3f',alpha_vec(i))],...
       'LineWidth',1.5,  'color', colors(i,:).^1.1);
    line(5) = plot(Idxs, iter_error2(:,i), ':', 'DisplayName',['\gamma = ', sprintf('%.3f',alpha_vec(i))],...
       'LineWidth',1.5,  'color', colors(i,:).^1.1);
    hold on
    hold on
end
ylabel("||\theta_k − \theta^*||^2")
ax = gca;
ax.YColor = 'k'; % set y-axis color to black

% Distance ratio lines
yyaxis left
for i = 3:3
    line(3) = plot(Idxs, ratio_seq(:,i),'-', 'DisplayName', ['\gamma = ' sprintf('%.3f',alpha_vec(i)) ' Tail-Avg'],...
        'LineWidth',1.5, 'color', colors(i,:).^1.1);
    hold on;
end

for i = 6:6
    line(6) = plot(Idxs, ratio_seq(:,i),'-', 'DisplayName', ['\gamma = ' sprintf('%.3f',alpha_vec(i)) ' Tail-Avg'],...
        'LineWidth',1.5, 'color', colors(i,:).^1.1);
    hold on;
end
ylabel("||D_k||^2 / ||D_0||^2")
ax = gca;
ax.YColor = 'k'; % set y-axis color to black
ylim([0, 3])
% set(gca,'ytick',[1e0, 1, 2, 3])

% Adjust Plot
lgnlabels = {};
for i = 3:3
   lgnlabels{1} = sprintf('||\\theta^{(1)}_k - \\theta^*||^2, \\gamma = %.2f',alpha_vec(i));
   lgnlabels{2} = sprintf('||\\theta^{(2)}_k - \\theta^*||^2, \\gamma = %.2f',alpha_vec(i));
end
for i = 6:6
   lgnlabels{4} = sprintf('||\\theta^{(1)}_k - \\theta^*||^2, \\gamma = %.2f',alpha_vec(i));
   lgnlabels{5} = sprintf('||\\theta^{(2)}_k - \\theta^*||^2, \\gamma = %.2f',alpha_vec(i));
end
for i = 3:3
   lgnlabels{3} = sprintf('||D_k||^2 / ||D_0||^2, \\gamma = %.2f',alpha_vec(i));
end
for i = 6:6
   lgnlabels{6} = sprintf('||D_k||^2 / ||D_0||^2, \\gamma = %.2f',alpha_vec(i));
end

hleg = legend('FontSize', fontsize-8, 'Location', 'northeast', 'FontName', 'Times New Roman');
legend(line, lgnlabels);

xlabel('Iteration k', 'FontSize', fontsize);
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
set(gca,'yscale','log');
% set(gca,'ytick',[1e-3, 1e-2, 1e-1, 1e0, 1e2])
ylim([1e-20, 1]);
set(gca,'xscale','log');
set(gca,'xtick',[1e1,1e2,1e3,1e4,1e5,1e6]);
xlim([1e0,K]);
set(gcf, 'Color', 'white');

pos = f.Position;
f.Position = [pos(1) pos(2)  800 450];

hold off

% exportgraphics(f, ['fig1_LSR_250103', '.png'], 'Resolution', 600)