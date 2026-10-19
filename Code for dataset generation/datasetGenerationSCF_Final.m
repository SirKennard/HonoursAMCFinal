% datasetGenerationSCF_v2.m
% takes the I/Q file from datasetGenerationIQ_v2.m and turns every frame into
% non-conjugate + conjugate SCF images, saved to .h5

clc; clear; close all;

%% Run options
MODE = 'run';        

PREVIEW_SNR = 10;      
PREVIEW_MOD = 1;       

USE_PARALLEL = 1;      % Use parallel but go back to serial if there's no Parallel Computing Toolbox
%% Input / output files
if ispc
    base_dir = 'D:\HonoursDataset456';
else
    base_dir = fullfile(fileparts(mfilename('fullpath')), 'HonoursDataset123');
end
% Change depending on which number of frames is used 

IQ_FILE = fullfile(base_dir, 'dataset_IQ_v2_2048.h5');
%IQ_FILE  = fullfile(base_dir, 'dataset_IQ_v2_4096.h5');
%IQ_FILE = fullfile(base_dir, 'dataset_IQ_v2_8192.h5');

% SCF file sits next to the I/Q one: 
[iq_dir, iq_name, iq_ext] = fileparts(IQ_FILE);
scf_name = strrep(iq_name, 'dataset_IQ', 'dataset_SCF');
if strcmp(scf_name, iq_name), scf_name = [iq_name '_SCF']; end   % To avoid writing over the I/Q file
SCF_FILE = fullfile(iq_dir, [scf_name iq_ext]);

%% SCF parameters
scf.T         = 64;           % window length in samples - spectral resolution fs/T
scf.Nfft      = 128;          % FFT size per window, extra over T is just zero-padding
scf.overlap   = scf.T - 1;    % hop = 1 sample, SSCA needs this
scf.N_smooth  = [];           % smoothing length in samples
scf.alpha_max = 50e3;       
scf.N_f_img   = 128;          % Rows = spectral freq f
scf.N_a_img   = 128;          % Columns = cycle freq alpha
scf.pool      = 'max';        % how fine points get squashed into a pixel: 'max' or 'mean'
scf.dB_range  = 40;           % keep 40 dB below the peak, anything lower -> 0

%% Mode
% 'run' and 'preview' both check the I/Q file is actually finished first
switch lower(MODE)
    case 'selftest'
        run_scf_selftest(scf, 4096);   % estimator check only, so it runs without an I/Q file
        return

    case 'preview'
        iq = check_iq_file(IQ_FILE);
        run_preview(IQ_FILE, scf_setup(scf, iq.N_samples, iq.fs), PREVIEW_SNR, PREVIEW_MOD);
        return

    case 'run'
        iq = check_iq_file(IQ_FILE);
        % check against theory first, before running, at this file's frame length
        if ~run_scf_selftest(scf, iq.N_samples)
            error('Self-test failed - no SCF file was made. Fix the failing check first.');
        end

    otherwise
        error('MODE must be ''run'', ''selftest'' or ''preview''');
end

% frame length and fs come from the I/Q file, since the SCF pixel map depends on them
N_samp_iq = iq.N_samples;
N_total   = iq.N_total;
iq_made   = iq.created;
iq_info   = iq.info;
scf_run   = scf_setup(scf, N_samp_iq, iq.fs);

%% Parallel pool
%Use if computer has it, else it takes more time. 
pct_available = license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
if USE_PARALLEL && ~pct_available
    warning('Parallel Computing Toolbox not available - falling back to serial execution.');
    USE_PARALLEL = 0;
end
if USE_PARALLEL && isempty(gcp('nocreate'))
    parpool;
end
if USE_PARALLEL
    scf_const = parallel.pool.Constant(scf_run);   
end

%% Create the h5 file
Nf     = scf_run.N_f_img;
Na     = scf_run.N_a_img;
Y_all  = h5read(IQ_FILE, '/Y');
N_mods = size(Y_all, 1);

fprintf('\n=== SCF DATASET -> %s ===\n', SCF_FILE);
fprintf('Total frames     : %d\n', N_total);
fprintf('SCF map size     : %dx%d, 2 channels (non-conjugate + conjugate)\n', Nf, Na);
fprintf('Parallel         : %d\n', USE_PARALLEL);
fprintf('Estimated uncompressed size: %.2f GB\n', N_total * (2*Nf*Na + 18) * 4 / 1e9);
print_scf_table(scf_run);

if exist(SCF_FILE, 'file'), delete(SCF_FILE); end
chunk_frames = min(256, N_total);
h5create(SCF_FILE, '/X_scf', [Nf, Na, N_total], ...
    'Datatype', 'single', 'ChunkSize', [Nf, Na, chunk_frames], 'Deflate', 4);
h5create(SCF_FILE, '/X_scf_conj', [Nf, Na, N_total], ...
    'Datatype', 'single', 'ChunkSize', [Nf, Na, chunk_frames], 'Deflate', 4);
h5create(SCF_FILE, '/Y', [N_mods, N_total], ...
    'Datatype', 'single', 'ChunkSize', [N_mods, chunk_frames], 'Deflate', 4);
h5create(SCF_FILE, '/Z', [1, N_total], ...
    'Datatype', 'single', 'ChunkSize', [1, chunk_frames], 'Deflate', 4);
h5create(SCF_FILE, '/imp', [4, N_total], ...
    'Datatype', 'single', 'ChunkSize', [4, chunk_frames], 'Deflate', 4);
% save the image axes too so we can plot a map later without working them out again
h5create(SCF_FILE, '/f_axis_hz', Nf);
h5write(SCF_FILE, '/f_axis_hz', scf_run.f_axis(:));
h5create(SCF_FILE, '/alpha_axis_hz', Na);
h5write(SCF_FILE, '/alpha_axis_hz', scf_run.alpha_axis(:));

%% Main loop
% read a batch of I/Q, SCF it in parallel, then one write per batch
buf_size  = min(1024, N_total);
N_batches = ceil(N_total / buf_size);
t_start   = tic;

for b = 1:N_batches
    batch_start = (b-1)*buf_size + 1;
    batch_end   = min(b*buf_size, N_total);
    cur_n       = batch_end - batch_start + 1;

    iq_batch     = h5read(IQ_FILE, '/X_iq', [1,1,batch_start], [2,N_samp_iq,cur_n]);
    X_scf_local  = zeros(Nf, Na, cur_n, 'single');
    X_conj_local = zeros(Nf, Na, cur_n, 'single');

    if USE_PARALLEL
        parfor i = 1:cur_n
            frame = iq_batch(:,:,i);
            rx    = double(frame(1,:)) + 1j*double(frame(2,:));
            [nc, c] = compute_scf_ssca(rx, scf_const.Value);
            X_scf_local(:,:,i)  = single(nc);
            X_conj_local(:,:,i) = single(c);
        end
    else
        for i = 1:cur_n
            frame = iq_batch(:,:,i);
            rx    = double(frame(1,:)) + 1j*double(frame(2,:));
            [nc, c] = compute_scf_ssca(rx, scf_run);
            X_scf_local(:,:,i)  = single(nc);
            X_conj_local(:,:,i) = single(c);
        end
    end

    h5write(SCF_FILE, '/X_scf',      X_scf_local,  [1,1,batch_start], [Nf,Na,cur_n]);
    h5write(SCF_FILE, '/X_scf_conj', X_conj_local, [1,1,batch_start], [Nf,Na,cur_n]);

    fprintf('  Batch %3d/%3d  (%d/%d frames)  %.1fs elapsed\n', ...
            b, N_batches, batch_end, N_total, toc(t_start));
end

% copy labels, SNR and impairments across so the SCF file works on its own
h5write(SCF_FILE, '/Y',   Y_all);
h5write(SCF_FILE, '/Z',   h5read(IQ_FILE, '/Z'));
h5write(SCF_FILE, '/imp', h5read(IQ_FILE, '/imp'));

%% Metadata
% carry over the I/Q file's settings, except the ones we write fresh below
for k = 1:numel(iq_info.Attributes)
    att = iq_info.Attributes(k);
    if ~any(strcmp(att.Name, {'complete', 'created', 'generator'}))
        h5writeatt(SCF_FILE, '/', att.Name, att.Value);
    end
end
s = scf_run;
h5writeatt(SCF_FILE, '/', 'generator',         'datasetGenerationSCF_v2.m');
h5writeatt(SCF_FILE, '/', 'source_iq_file',    [iq_name iq_ext]);
h5writeatt(SCF_FILE, '/', 'source_iq_created', iq_made);   % lets us match it back to the exact I/Q file
h5writeatt(SCF_FILE, '/', 'SCF_estimator',     'SSCA (Roberts, Brown & Loomis 1991), non-conjugate + conjugate');
h5writeatt(SCF_FILE, '/', 'SCF_axes',          'MATLAB [f, alpha, frame] -> h5py (frame, alpha, f)');
h5writeatt(SCF_FILE, '/', 'SCF_scaling',       'dB below the non-conjugate peak (both channels), clipped to [-dB_range, 0], mapped to [0, 1]');
h5writeatt(SCF_FILE, '/', 'SCF_T',             int32(s.T));
h5writeatt(SCF_FILE, '/', 'SCF_Nfft',          int32(s.Nfft));
h5writeatt(SCF_FILE, '/', 'SCF_overlap',       int32(s.overlap));
h5writeatt(SCF_FILE, '/', 'SCF_hop',           int32(s.hop));
h5writeatt(SCF_FILE, '/', 'SCF_N_smooth',      int32(s.N_smooth));
h5writeatt(SCF_FILE, '/', 'SCF_strip_fft',     int32(s.Ns));
h5writeatt(SCF_FILE, '/', 'SCF_df_res_hz',     s.df_res);
h5writeatt(SCF_FILE, '/', 'SCF_df_img_hz',     s.df_img);
h5writeatt(SCF_FILE, '/', 'SCF_dalpha_res_hz', s.dalpha_res);
h5writeatt(SCF_FILE, '/', 'SCF_dalpha_img_hz', s.dalpha_img);
h5writeatt(SCF_FILE, '/', 'SCF_alpha_max_hz',  s.alpha_max);
h5writeatt(SCF_FILE, '/', 'SCF_pool',          s.pool);
h5writeatt(SCF_FILE, '/', 'SCF_dB_range',      s.dB_range);
h5writeatt(SCF_FILE, '/', 'SCF_TB',            s.TB);
h5writeatt(SCF_FILE, '/', 'SCF_window',        'hamming periodic (channeliser), hamming (smoothing taper)');
h5writeatt(SCF_FILE, '/', 'USE_PARALLEL_SCF',  int32(USE_PARALLEL));
h5writeatt(SCF_FILE, '/', 'created',           char(datetime('now')));
h5writeatt(SCF_FILE, '/', 'complete',          int32(1));

fprintf('\nDone. %.1fs  |  %s  (%.2f GB)\n', toc(t_start), SCF_FILE, dir(SCF_FILE).bytes/1e9);
fprintf('  X_scf      shape : [%d x %d x %d]\n', Nf, Na, N_total);
fprintf('  X_scf_conj shape : [%d x %d x %d]\n', Nf, Na, N_total);
fprintf('  made from        : %s (created %s)\n', [iq_name iq_ext], iq_made);


%% Local functions

function iq = check_iq_file(IQ_FILE)
    % Check the I/Q file exists, is complete, then read info
    if ~exist(IQ_FILE, 'file')
        error('Can''t find the I/Q dataset %s - run datasetGenerationIQ_v2.m first.', IQ_FILE);
    end
    info = h5info(IQ_FILE);
    ds_names = {};  att_names = {};
    if ~isempty(info.Datasets),   ds_names  = {info.Datasets.Name};   end
    if ~isempty(info.Attributes), att_names = {info.Attributes.Name}; end
    missing = setdiff({'X_iq', 'Y', 'Z', 'imp'}, ds_names);
    if ~isempty(missing)
        error('%s has no %s - is it really from datasetGenerationIQ_v2.m?', IQ_FILE, strjoin(missing, ', '));
    end
    if ~any(strcmp(att_names, 'complete'))
        error('%s has no ''complete'' flag, so the I/Q run never finished - run datasetGenerationIQ_v2.m again.', IQ_FILE);
    end

    % Read frame size, sample rate and creation date, then print
    ds = h5info(IQ_FILE, '/X_iq');
    iq.info      = info;
    iq.N_samples = ds.Dataspace.Size(2);
    iq.N_total   = ds.Dataspace.Size(3);
    iq.fs        = double(h5readatt(IQ_FILE, '/', 'fs_hz'));
    iq.created   = h5readatt(IQ_FILE, '/', 'created');
    fprintf('\nRaw I/Q dataset  : %s\n', IQ_FILE);
    fprintf('  made %s  |  %d frames x %d samples  |  fs = %.0f kHz\n', ...
        iq.created, iq.N_total, iq.N_samples, iq.fs/1e3);
end

% SSCA: compute non-conjugate and conjugate SCF of one frame
function [img_nc, img_c, S_nc, S_c] = compute_scf_ssca(rx, scf)
    rx = rx(:);
    rx = rx / sqrt(mean(abs(rx).^2));     % normalise to unit power

    % Sliding windowed FFTs, phase-corrected to a common time reference
    starts = (0:scf.M-1) * scf.hop;
    idx    = (1:scf.T).' + starts;
    X = fft(rx(idx) .* scf.win, scf.Nfft, 1);
    X = X .* exp(-1j*2*pi*(0:scf.Nfft-1).' * starts / scf.Nfft);

    % Multiply by centre sample: conjugated (non-conj) or not (conj)
    xc = rx(starts + scf.T/2 + 1).';
    Z_nc = X .* conj(xc);
    Z_c  = X .* xc;

    % Strip FFT along time, then pool onto the (f, alpha) grid
    S_nc = pool_points(strip_fft(Z_nc, scf), scf);
    S_c  = pool_points(strip_fft(Z_c,  scf), scf);

    % Scale both to the non-conj peak, keeping relative strength
    ref    = max(S_nc(:));
    img_nc = to_image(S_nc, ref, scf.dB_range);
    img_c  = to_image(S_c,  ref, scf.dB_range);
end


% FFT each channel along time, averaged over smoothing blocks
% Shorter blocks give coarser alpha resolution (fs / N_smooth)
function S = strip_fft(Z, scf)
    S = zeros(scf.Nfft, scf.Ns);
    for k = 1:scf.n_seg
        cols = (k-1)*scf.N_smooth + 1 : k*scf.N_smooth;
        S = S + abs(fft(Z(:, cols) .* scf.g, scf.Ns, 2));
    end
    S = S / (scf.n_seg * sum(scf.g));
end

% Combine fine SCF points into image pixels using max or mean
function img = pool_points(S, scf)
    vals = S(scf.keep);
    switch scf.pool
        case 'max'
            img = accumarray(scf.pix, vals, [scf.N_f_img * scf.N_a_img, 1], @max, 0);
        case 'mean'
            img = accumarray(scf.pix, vals, [scf.N_f_img * scf.N_a_img, 1]) ./ max(scf.count, 1);
        otherwise
            error('scf.pool must be ''max'' or ''mean''');
    end
    img = reshape(img, scf.N_f_img, scf.N_a_img);
end

% Convert to dB and scale so peak = 1, -dB_range = 0
function img = to_image(A, ref, dB_range)
    img = 1 + 10*log10(A / ref + 1e-12) / dB_range;
    img = min(max(img, 0), 1);            % clip values to the 0-1 range
end

% Precompute settings shared by every frame (windows, axes, pixel map)
function scf = scf_setup(scf, N_samples, fs)
    scf.fs        = fs;
    scf.N_samples = N_samples;
    scf.hop       = scf.T - scf.overlap;
    if mod(scf.T, 2) || mod(scf.Nfft, 2) || scf.Nfft < scf.T
        error('scf.T and scf.Nfft need to be even, with Nfft >= T');
    end
    if scf.hop ~= 1
        warning('SSCA needs hop = 1 (overlap = T-1). hop = %d will fold the alpha axis.', scf.hop);
    end

    % Work out number of windows, smoothing blocks and strip FFT size
    scf.M = floor((N_samples - scf.T) / scf.hop) + 1;
    if isempty(scf.N_smooth), scf.N_smooth = scf.M; end
    scf.N_smooth = min(scf.N_smooth, scf.M);
    scf.n_seg    = floor(scf.M / scf.N_smooth);
    scf.Ns       = 2^nextpow2(scf.N_smooth);

    % Create the analysis window and smoothing taper
    scf.win = hamming(scf.T, 'periodic');   % peak lands on the centre sample
    scf.g   = hamming(scf.N_smooth).';      % reduces leakage along alpha

    % Calculate resolutions and spacings for the parameter table
    scf.df_res       = fs / scf.T;
    scf.df_strip     = fs / scf.Nfft;
    scf.df_img       = fs / scf.N_f_img;
    scf.dalpha_res   = fs / (scf.N_smooth * scf.hop);
    scf.dalpha_strip = fs / (scf.Ns * scf.hop);
    scf.dalpha_img   = 2 * scf.alpha_max / scf.N_a_img;
    scf.pool_factor  = scf.dalpha_img / scf.dalpha_strip;
    scf.smooth_ms    = 1e3 * scf.N_smooth * scf.hop / fs;
    scf.TB           = scf.N_smooth * scf.hop / scf.T;    % should be >> 1

    % Map every fine (f, alpha) point to its nearest image pixel
    fk    = (mod((0:scf.Nfft-1).' + scf.Nfft/2, scf.Nfft) - scf.Nfft/2) / scf.Nfft;
    q     = (mod((0:scf.Ns-1) + scf.Ns/2, scf.Ns) - scf.Ns/2) / (scf.Ns * scf.hop);
    alpha = fk + q;
    f     = (fk - q) / 2;
    i_f = floor(f * scf.N_f_img + 0.5) + scf.N_f_img/2 + 1;
    i_a = floor(alpha * fs / scf.dalpha_img + 0.5) + scf.N_a_img/2 + 1;
    ok  = i_f >= 1 & i_f <= scf.N_f_img & i_a >= 1 & i_a <= scf.N_a_img;
    scf.keep  = find(ok);
    scf.pix   = (i_a(ok) - 1) * scf.N_f_img + i_f(ok);
    scf.count = accumarray(scf.pix, 1, [scf.N_f_img * scf.N_a_img, 1]);

    % Build the frequency and cycle-frequency axes in Hz
    scf.f_axis     = ((1:scf.N_f_img) - scf.N_f_img/2 - 1) * fs / scf.N_f_img;
    scf.alpha_axis = ((1:scf.N_a_img) - scf.N_a_img/2 - 1) * scf.dalpha_img;
end

% Print a table of SCF settings and their resulting resolutions
function print_scf_table(scf)
    fprintf('\n=== SCF (SSCA) parameters ===\n');
    fprintf('  %-22s %-16s %s\n', 'parameter', 'value', 'what it gives');
    fprintf('  %-22s %-16s %s\n', 'analysis window T', sprintf('%d samples', scf.T), ...
        sprintf('spectral resolution fs/T = %.3f kHz', scf.df_res/1e3));
    fprintf('  %-22s %-16s %s\n', 'FFT size Nfft', sprintf('%d', scf.Nfft), ...
        sprintf('channel spacing fs/Nfft = %.4f kHz (%gx zero-padding)', scf.df_strip/1e3, scf.Nfft/scf.T));
    fprintf('  %-22s %-16s %s\n', 'window overlap', sprintf('%d samples', scf.overlap), ...
        sprintf('hop = %d sample(s)', scf.hop));
    fprintf('  %-22s %-16s %s\n', 'delta f', sprintf('%.4f kHz', scf.df_img/1e3), ...
        sprintf('image rows, f = %.1f ... %.1f kHz', scf.f_axis(1)/1e3, scf.f_axis(end)/1e3));
    fprintf('  %-22s %-16s %s\n', 'delta alpha', sprintf('%.1f Hz', scf.dalpha_img), ...
        sprintf('image columns, alpha = %.1f ... %.1f kHz (fine estimate: %.1f Hz)', ...
        scf.alpha_axis(1)/1e3, scf.alpha_axis(end)/1e3, scf.dalpha_res));
    fprintf('  %-22s %-16s %s\n', 'interpolation', scf.pool, ...
        sprintf('%.0f fine alpha bins pooled into each column', scf.pool_factor));
    fprintf('  %-22s %-16s %s\n', 'smoothing duration', sprintf('%d samples', scf.N_smooth), ...
        sprintf('%.1f ms, %d block(s), strip FFT size %d', scf.smooth_ms, scf.n_seg, scf.Ns));
    fprintf('  %-22s %-16s %s\n', 'reliability dt*df', sprintf('%.0f', scf.TB), ...
        'smoothing time x spectral resolution, wants to be >> 1');
    fprintf('  %-22s %-16s %s\n\n', 'dynamic range', sprintf('%g dB', scf.dB_range), ...
        'both channels, dB below the non-conjugate peak');
end

%% Preview of data
% Print SCF settings and preview SCF images of real frames
% from the I/Q file, to check settings before the full run
function run_preview(IQ_FILE, scf, snr, mod_detail)
    % Print the SCF parameter table and create the output folder
    print_scf_table(scf);
    out_dir = 'scf_v2_preview';
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    % Load labels, SNRs, impairments and modulation info from file
    Y   = h5read(IQ_FILE, '/Y');
    Z   = h5read(IQ_FILE, '/Z');
    imp = h5read(IQ_FILE, '/imp');
    [~, lab] = max(Y, [], 1);
    mods = strsplit(h5readatt(IQ_FILE, '/', 'mod_list'), ',');
    sps  = double(h5readatt(IQ_FILE, '/', 'sps'));
    N_mods     = numel(mods);
    mod_detail = min(mod_detail, N_mods);

    % Compute SCFs for the first frame of each class at chosen SNR
    NC = zeros(scf.N_f_img, scf.N_a_img, N_mods);
    CJ = NC;
    picked = zeros(1, N_mods);
    for m = 1:N_mods
        n = find(lab == m & Z == snr, 1);
        if isempty(n)
            error('No %s frame at %g dB in the I/Q file - pick a PREVIEW_SNR from its SNR list.', mods{m}, snr);
        end
        picked(m) = n;
        iq = h5read(IQ_FILE, '/X_iq', [1 1 n], [2 scf.N_samples 1]);
        rx_m = double(iq(1,:)) + 1j*double(iq(2,:));
        [NC(:,:,m), CJ(:,:,m)] = compute_scf_ssca(rx_m, scf);
        if m == mod_detail, rx = rx_m; end
    end

    % Convert cycle-frequency and frequency axes from Hz to kHz
    fa = scf.alpha_axis / 1e3;
    ff = scf.f_axis / 1e3;

    % Plot a 4x4 grid of all classes for each SCF type
    names = {'non-conjugate SCF', 'conjugate SCF'};
    files = {'preview_nonconj.png', 'preview_conj.png'};
    for ch = 1:2
        fig = figure('Color', 'w', 'Position', [40 40 1300 1100]);
        tl  = tiledlayout(4, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
        for m = 1:N_mods
            nexttile;
            if ch == 1, A = NC(:,:,m); else, A = CJ(:,:,m); end
            imagesc(fa, ff, A, [0 1]); axis xy;
            title(sprintf('%s  (frame %d, CFO %.1f kHz)', mods{m}, picked(m), imp(1, picked(m))/1e3), 'FontSize', 9);
            if mod(m-1, 4) == 0, ylabel('f (kHz)'); end
            if m > N_mods - 4, xlabel('\alpha (kHz)'); end
        end

        % Add a shared colourbar, overall title, then save as PNG
        colormap(fig, parula);
        cb = colorbar; cb.Layout.Tile = 'east';
        cb.Label.String = sprintf('0 = %g dB below the non-conjugate peak, 1 = peak', scf.dB_range);
        title(tl, sprintf('%s of frames from the I/Q file, %g dB SNR  |  T=%d, Nfft=%d, hop=%d, smoothing %.1f ms, pool=%s', ...
            names{ch}, snr, scf.T, scf.Nfft, scf.hop, scf.smooth_ms, scf.pool));
        exportgraphics(fig, fullfile(out_dir, files{ch}), 'Resolution', 150);
    end
   

    % Get the selected frame, its symbol rate and CFO
    n   = picked(mod_detail);
    Rs  = scf.fs / sps * imp(2, n);
    cfo = imp(1, n);

    % Create a 2x3 figure layout for the diagnostic plots
    fig = figure('Color', 'w', 'Position', [60 60 1300 800]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Plot the first 400 I Q samples over time
    nexttile;
    t_ms = (0:399) / scf.fs * 1e3;
    plot(t_ms, real(rx(1:400)), 'b', t_ms, imag(rx(1:400)), 'r');
    title(sprintf('%s I/Q (frame %d), %g dB', mods{mod_detail}, n, snr)); xlabel('time (ms)'); legend('I', 'Q'); grid on;

    % Plot the power spectral density using Welch's method
    nexttile;
    [pxx, fp] = pwelch(rx, hamming(512), 256, 1024, scf.fs, 'centered');
    plot(fp/1e3, 10*log10(pxx), 'k'); grid on;
    title('PSD'); xlabel('f (kHz)'); ylabel('dB/Hz');
   
    % Display impairment values and where SCF features should appear

    nexttile; axis off;
    text(0, 0.5, {sprintf('CFO = %.0f Hz', cfo), sprintf('symbol rate = %.0f Hz', Rs), ...
        sprintf('timing offset = %.2f samples', imp(3, n)), sprintf('K = %.1f', imp(4, n)), '', ...
        'non-conj features: \alpha = 0, \pm symbol rate', 'conj features: \alpha = 2 CFO, 2 CFO \pm symbol rate', ...
        '(the conj ones only exist for BPSK / PAM4 / AM / FSK types)'}, 'FontSize', 10);
    
    % Plot non-conjugate SCF with expected feature lines overlaid
    nexttile;
    imagesc(fa, ff, NC(:,:,mod_detail), [0 1]); axis xy; colorbar;
    xline([-Rs 0 Rs]/1e3, 'w:'); title('non-conjugate SCF'); xlabel('\alpha (kHz)'); ylabel('f (kHz)');

    % Plot conjugate SCF with expected feature lines overlaid
    nexttile;
    imagesc(fa, ff, CJ(:,:,mod_detail), [0 1]); axis xy; colorbar;
    xline((2*cfo + [-Rs 0 Rs])/1e3, 'w:'); title('conjugate SCF'); xlabel('\alpha (kHz)'); ylabel('f (kHz)');

    % Plot peak SCF value at each cycle frequency
    nexttile;
    plot(fa, max(NC(:,:,mod_detail), [], 1), 'b', fa, max(CJ(:,:,mod_detail), [], 1), 'r'); grid on;
    xline([-Rs 0 Rs]/1e3, 'b:'); xline((2*cfo + [-Rs 0 Rs])/1e3, 'r:');
    legend('non-conj', 'conj', 'Location', 'southoutside', 'Orientation', 'horizontal');
    title('cycle-frequency profile (max over f)'); xlabel('\alpha (kHz)');

    % Save the figure as a PNG to the output folder
    exportgraphics(fig, fullfile(out_dir, 'preview_detail.png'), 'Resolution', 150);
    fprintf('Preview figures saved to %s/\n', out_dir);
end


%% Self-test
% Ensure the SCF matches theory before we run it on the whole dataset
function all_ok = run_scf_selftest(scf_settings, N_frame)
    fprintf('=== SCF SELF-TEST (frame length %d) ===\n', N_frame);
    fs = 200e3; sps = 8; rolloff = 0.35;
    Rs = fs / sps;
    n_fail = 0;
    rng(1);

    % long clean signal (8 frames' worth) so estimator noise doesn't blur the checks,
    % capped so the test still takes seconds on the 8192 set
    N_long = min(8 * N_frame, 32768);
    S = scf_setup(scf_settings, N_long, fs);
    col    = @(a) find(abs(S.alpha_axis - a) == min(abs(S.alpha_axis - a)), 1);
    inband = abs(S.f_axis) <= 10e3;
    dB     = @(x) 10*log10(x);
    ridge  = @(A, c) mean(A(inband, c));        % average down one alpha column, in-band only

    function report(name, value, ok, expect)
        if ok, tag = 'PASS'; else, tag = 'FAIL'; n_fail = n_fail + 1; end
        fprintf('  [%s] %-46s %8.2f   (%s)\n', tag, name, value, expect);
    end

    % BPSK conj is the same as non-conj
    x = test_signal('BPSK', N_long, sps, rolloff) * exp(1j*2*pi*rand);
    [~, ~, Snc, Sc] = compute_scf_ssca(x, S);
    v = dB(max(Snc(:, col(Rs))) / max(Snc(:, col(0))));
    report('BPSK non-conj at alpha = Rs (dB)', v, v > -7 && v < -1.5, 'theory -3, the T window smooths ~1.5 off');
    v = dB(ridge(Sc, col(0)) / ridge(Snc, col(0)));
    report('BPSK conj ridge at alpha = 0 (dB)', v, abs(v) < 1, 'theory 0');
    v = dB(max(Sc(:, col(Rs))) / max(Sc(:, col(0))));
    report('BPSK conj at alpha = Rs (dB)', v, v > -7 && v < -1.5, 'theory -3, same smoothing');

    % PAM4 should have conj features, QPSK and 16QAM shouldn't
    x = test_signal('PAM4', N_long, sps, rolloff) * exp(1j*2*pi*rand);
    [~, ~, Snc, Sc] = compute_scf_ssca(x, S);
    v = dB(ridge(Sc, col(0)) / ridge(Snc, col(0)));
    report('PAM4 conj ridge at alpha = 0 (dB)', v, abs(v) < 1, 'theory 0');
    for kind = {'QPSK', '16QAM'}
        x = test_signal(kind{1}, N_long, sps, rolloff) * exp(1j*2*pi*rand);
        [~, ~, Snc, Sc] = compute_scf_ssca(x, S);
        v = dB(max(mean(Sc(inband, :), 1)) / ridge(Snc, col(0)));
        report(sprintf('%s strongest conj ridge anywhere (dB)', kind{1}), v, v < -5, 'theory: none, just noise');
    end

    % with a carrier offset, non-conj should stay at alpha = 0
    % and conj should move to alpha = 2*CFO
    cfo = 3300;
    n = (0:N_long-1).';
    x = test_signal('BPSK', N_long, sps, rolloff) .* exp(1j*(2*pi*cfo*n/fs + 2*pi*rand));
    [~, ~, Snc, Sc] = compute_scf_ssca(x, S);
    [~, c1] = max(mean(Snc(inband, :), 1));
    [~, c2] = max(mean(Sc(inband, :), 1));
    v = S.alpha_axis(c1);
    report('CFO 3.3 kHz: non-conj ridge at alpha (Hz)', v, abs(v) <= S.dalpha_img, 'expect 0');
    v = S.alpha_axis(c2);
    report('CFO 3.3 kHz: conj ridge at alpha (Hz)', v, abs(v - 2*cfo) <= S.dalpha_img, 'expect 6600');

    % white noise shouldn't have any lines off alpha = 0 
    w = (randn(N_long, 1) + 1j*randn(N_long, 1)) / sqrt(2);
    [~, ~, Snc] = compute_scf_ssca(w, S);
    far = abs(S.alpha_axis) > 2e3;
    v = dB(max(mean(Snc(:, far), 1)) / mean(Snc(:, col(0))));
    report('white noise: strongest column off alpha=0 (dB)', v, v < -6, 'expect well below 0');

    % check the alpha axis is in the right units 
    Ng = ceil(N_long * 1.03) + 64;
    x  = test_signal('BPSK', Ng, sps, rolloff);
    xj = interp1((0:Ng-1).', x, (0:N_long-1).' * 1.02, 'spline');
    [~, ~, Snc] = compute_scf_ssca(xj, S);
    pos = S.alpha_axis > 15e3;
    [~, c3] = max(max(Snc(:, pos), [], 1));
    ap = S.alpha_axis(pos);
    v = ap(c3);
    report('symbols x1.02: symbol-rate feature at (Hz)', v, abs(v - 1.02*Rs) <= S.dalpha_img, 'expect 25500');

    % a frame the size this dataset actually uses comes out the right size,
    % no NaN/Inf, all in [0,1]
    S4 = scf_setup(scf_settings, N_frame, fs);
    [a, c] = compute_scf_ssca(awgn(test_signal('QPSK', N_frame, sps, rolloff), 0, 'measured'), S4);
    ok = isequal(size(a), [S4.N_f_img S4.N_a_img]) && isequal(size(c), size(a)) ...
         && all(isfinite([a(:); c(:)])) && min([a(:); c(:)]) >= 0 && max([a(:); c(:)]) <= 1;
    report(sprintf('%d-sample frame: sized, finite, in [0,1]', N_frame), double(ok), ok, '1 = yes');

    all_ok = (n_fail == 0);
    if all_ok
        fprintf('=== ALL PASSED ===\n');
    else
        fprintf('=== %d CHECK(S) FAILED - fix before making the SCF file ===\n', n_fail);
    end
end

function x = test_signal(kind, N, sps, rolloff)

    % clean RRC test signal, only used by the self-test (real data comes from the I/Q file)
    n_sym = ceil(N/sps) + 16;
    switch kind
        case 'BPSK',  a = 2*randi([0 1], n_sym, 1) - 1;
        case 'PAM4',  a = (2*randi([0 3], n_sym, 1) - 3) / sqrt(5);
        case 'QPSK',  a = exp(1j*(pi/4 + pi/2*randi([0 3], n_sym, 1)));
        case '16QAM', a = ((2*randi([0 3], n_sym, 1) - 3) + 1j*(2*randi([0 3], n_sym, 1) - 3)) / sqrt(10);
    end
    h = rcosdesign(rolloff, 6, sps, 'sqrt');
    x = filter(h, 1, upsample(a, sps));
    x = x(3*sps + 1 : 3*sps + N);                 % skip the filter delay (span*sps/2)
    x = x / sqrt(mean(abs(x).^2));
end
