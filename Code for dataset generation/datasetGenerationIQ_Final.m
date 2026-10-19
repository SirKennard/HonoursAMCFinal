% datasetGenerationIQ_Final_2048Frames.m

clc; clear; close all;

%% Run options
MODE   = 'run';     % 'run' = self-test (Check all contents and modulations are correct) then generate
SCOPE  = 'full';    % 'full' = all 13 mods, 'individual' = only MOD_ID
MOD_ID = 1;
% 1=BPSK  2=QPSK  3=8PSK  4=16QAM  5=64QAM   6=PAM4   7=GFSK
% 8=CPFSK 9=2FSK  10=4FSK 11=B-FM  12=AM-DSB 13=AM-SSB

USE_PARALLEL = 1;   
                    
DEFLATE = 4; %To compress the HDF file 

%% Dataset size
N_frames_per_snr = 1024;
SNR_dB_vec       = -20:2:30;
BASE_SEED        = 2026;

%% Output folder
if ispc
    base_dir = 'D:HonoursDataset456';
else
    base_dir = fullfile(fileparts(mfilename('fullpath')), 'HonoursDataset123');
end

%% Signal + impairment parameters
P.N_samples     = 2048;                % 2048, 4096 and 8196 frames. Change
P.sps           = 8;
P.span          = 6;
P.rolloff       = 0.35;
P.fs            = 200e3;
P.Rs            = P.fs / P.sps;
% GFSK and CPFSK are same family, both h = 0.5, so pulse length is the only thing telling
% them apart. Hence CPMModulator defaults to L = 1 which is basically rectangular,
% so GFSK would just be CPFSK - hence L = 4.
P.gfsk_BT       = 0.3;
P.gfsk_L        = 4;                   % pulse length in symbols
P.gfsk_h        = 0.5;
P.cpfsk_h       = 0.5;

%2FSK at h = 0.5 is literally the same signal as CPFSK at h = 0.5.
% Hence bump 2FSK up to h = 1.
P.fsk2_sep      = P.fs / P.sps;        % 25 kHz, h = 1
P.fsk4_sep      = P.fs / (4*P.sps);    % 6.25 kHz, h = 0.25

P.cfo_ppm_max   = 5;
P.f_carrier     = 902e6;
P.Rs_var_frac   = 0.02;
P.K_min         = 0;
P.K_max         = 10;
P.doppler_hz    = 4;
P.path_delays   = [0, 1.8, 3.4] / P.fs;
P.path_gains_dB = [0, -2, -10];

P.USE_SYMBOL_RATE_VAR = 1;
P.USE_PHASE_OFFSET    = 1;
P.USE_CFO             = 1;
P.USE_TIMING_OFFSET   = 1;
P.USE_FADING          = 1;

% all the digital signal have a symbol clock, hence jitter and timing offset
P.has_timing = {'psk','qam','pam','gfsk','cpfsk','fsk'};
P.seed       = BASE_SEED;

%% Modulation table
mod_table = {
    1,  'BPSK',    'psk',    2;
    2,  'QPSK',    'psk',    4;
    3,  '8PSK',    'psk',    8;
    4,  '16QAM',   'qam',    16;
    5,  '64QAM',   'qam',    64;
    6,  'PAM4',    'pam',    4;
    7,  'GFSK',    'gfsk',   2;
    8,  'CPFSK',   'cpfsk',  2;
    9,  '2FSK',    'fsk',    2;
    10, '4FSK',    'fsk',    4;
    11, 'B-FM',    'fm',     0;
    12, 'AM-DSB',  'am_dsb', 0;
    13, 'AM-SSB',  'am_ssb', 0;
};
N_mods_total = size(mod_table, 1);

%% Running the file - Mode / scope / filename

% Run self-test only, or self-test then full generation
switch lower(MODE)
    case 'selftest'
        run_iq_selftest(P, mod_table);
        return
    case 'run'
        if ~run_iq_selftest(P, mod_table)
            error('Self-test failed - nothing generated. Fix the failing check first.');
        end
    otherwise
        error('MODE must be ''run'' or ''selftest''');
end

% Choose all modulations or a single one by its ID
switch lower(SCOPE)
    case 'full'
        mod_ids  = 1:N_mods_total;
        name_tag = '';
        fprintf('=== SCOPE: FULL (%d modulations) ===\n', N_mods_total);
    case 'individual'
        if MOD_ID < 1 || MOD_ID > N_mods_total
            error('MOD_ID must be between 1 and %d', N_mods_total);
        end
        mod_ids  = MOD_ID;
        name_tag = ['_' mod_table{MOD_ID,2}];
        P.seed   = BASE_SEED + MOD_ID;   % unique seed so each class gets different random draws
        fprintf('=== SCOPE: INDIVIDUAL - %s (ID=%d) ===\n', mod_table{MOD_ID,2}, MOD_ID);
    otherwise
        error('SCOPE must be ''full'' or ''individual''');
end

% Build save path with frame length so files never overwrite
SAVE_PATH = fullfile(base_dir, sprintf('dataset_IQ_v2_%d%s.h5', P.N_samples, name_tag));

% Output folder i
if ~exist(base_dir, 'dir')
    mkdir(base_dir);
    fprintf('Created folder: %s\n', base_dir);
end

%% Parallel Processing
% frames don't depend on each other so a whole batch can be built at once.
%Also saves time with generation
if USE_PARALLEL && ~(license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel')))
    warning('No Parallel Computing Toolbox - running serial.');
    USE_PARALLEL = 0;
end
if USE_PARALLEL
    if isempty(gcp('nocreate')), parpool; end
    P_const = parallel.pool.Constant(P);   
end

%% Sizes
N_mods  = numel(mod_ids);
N_snr   = numel(SNR_dB_vec);
N_total = N_mods * N_snr * N_frames_per_snr;
N_samp  = P.N_samples;

fprintf('Modulations      : %s\n', strjoin(mod_table(mod_ids,2)', ', '));
fprintf('SNR levels       : %d  (%d to %d dB)\n', N_snr, min(SNR_dB_vec), max(SNR_dB_vec));
fprintf('Frames per SNR   : %d\n', N_frames_per_snr);
fprintf('Total frames     : %d\n', N_total);
fprintf('Frame length     : %d samples (%.1f ms)\n', N_samp, 1e3*N_samp/P.fs);
fprintf('Uncompressed     : %.2f GB\n', N_total*(2*N_samp + 18)*4/1e9);
fprintf('Save path        : %s\n\n', SAVE_PATH);

%% Create the h5 file
if exist(SAVE_PATH, 'file'), delete(SAVE_PATH); end

% 2048: twice as many frames per chunk, so a chunk stays about 8 MB like the
% 4096 file. Big chunks are slow to read back one frame at a time.
chunk_frames = min(512, N_total);
h5create(SAVE_PATH, '/X_iq', [2, N_samp, N_total], 'Datatype', 'single', ...
    'ChunkSize', [2, N_samp, chunk_frames], 'Deflate', DEFLATE);
h5create(SAVE_PATH, '/Y', [N_mods, N_total], 'Datatype', 'single', ...
    'ChunkSize', [N_mods, chunk_frames], 'Deflate', 4);
h5create(SAVE_PATH, '/Z', [1, N_total], 'Datatype', 'single', ...
    'ChunkSize', [1, chunk_frames], 'Deflate', 4);
h5create(SAVE_PATH, '/imp', [4, N_total], 'Datatype', 'single', ...
    'ChunkSize', [4, chunk_frames], 'Deflate', 4);

%% Shuffle
% shuffle so a naive split still will include every class
all_local_mod = repelem((1:N_mods)', N_snr * N_frames_per_snr);
all_snrs      = repmat(repelem(SNR_dB_vec', N_frames_per_snr), N_mods, 1);
shuffle       = randperm(RandStream('mrg32k3a', 'Seed', P.seed), N_total);
all_local_mod = all_local_mod(shuffle);
all_snrs      = all_snrs(shuffle);

%% Main loop
%Generate frames in batches of up to 1024
buf_size  = min(1024, N_total);
N_batches = ceil(N_total / buf_size);
t_start   = tic;

% Get the frame indices covered by this batch
for b = 1:N_batches
    batch_start = (b-1)*buf_size + 1;
    batch_end   = min(b*buf_size, N_total);
    cur_n       = batch_end - batch_start + 1;
    batch_idx   = batch_start:batch_end;
   
    % Look up modulation type, order and SNR for each frame
    batch_local_mod = all_local_mod(batch_idx).';
    batch_snr       = all_snrs(batch_idx).';
    batch_global_id = mod_ids(batch_local_mod);
    batch_mod_type  = mod_table(batch_global_id, 3);
    batch_mod_M     = cell2mat(mod_table(batch_global_id, 4));

    % Preallocate for I/Q samples and impairment values
    X_buf = zeros(2, N_samp, cur_n, 'single');
    M_buf = zeros(4, cur_n, 'single');

    % Generate each frame, if parallel if enabled
    if USE_PARALLEL
        parfor i = 1:cur_n
            [iq, imp] = make_iq_frame(batch_mod_type{i}, batch_mod_M(i), ...
                                      batch_snr(i), batch_idx(i), P_const.Value);
            X_buf(:,:,i) = iq;
            M_buf(:,i)   = imp;
        end
    else
        for i = 1:cur_n
            [iq, imp] = make_iq_frame(batch_mod_type{i}, batch_mod_M(i), ...
                                      batch_snr(i), batch_idx(i), P);
            X_buf(:,:,i) = iq;
            M_buf(:,i)   = imp;
        end
    end

    % Build one-hot class labels and SNR values for the batch
    Y_buf = single(batch_local_mod == (1:N_mods)');
    Z_buf = single(batch_snr);

    % Write the batch into its slot in the HDF5 file
    h5write(SAVE_PATH, '/X_iq', X_buf, [1, 1, batch_start], [2, N_samp, cur_n]);
    h5write(SAVE_PATH, '/Y',    Y_buf, [1, batch_start],    [N_mods, cur_n]);
    h5write(SAVE_PATH, '/Z',    Z_buf, [1, batch_start],    [1, cur_n]);
    h5write(SAVE_PATH, '/imp',  M_buf, [1, batch_start],    [4, cur_n]);

    % Progress and estimated time remaining
    el = toc(t_start);
    fprintf('  %3.0f%%  (%d/%d)  %.0fs elapsed, ~%.0fs remaining\n', ...
            100*batch_end/N_total, batch_end, N_total, el, ...
            el*(N_total - batch_end)/batch_end);
end

%% Metadata

% Save dataset settings as file attributes for SCF
h5writeatt(SAVE_PATH, '/', 'mod_list',          strjoin(mod_table(mod_ids,2)', ','));
h5writeatt(SAVE_PATH, '/', 'mod_ids',           int32(mod_ids));
h5writeatt(SAVE_PATH, '/', 'scope',             SCOPE);
h5writeatt(SAVE_PATH, '/', 'generator',         'datasetGenerationIQ_Final_2048Frames.m');
h5writeatt(SAVE_PATH, '/', 'N_total',           int32(N_total));
h5writeatt(SAVE_PATH, '/', 'N_samples',         int32(N_samp));
h5writeatt(SAVE_PATH, '/', 'frame_ms',          1e3*N_samp/P.fs);
h5writeatt(SAVE_PATH, '/', 'N_frames_per_snr',  int32(N_frames_per_snr));
h5writeatt(SAVE_PATH, '/', 'N_snr_levels',      int32(N_snr));
h5writeatt(SAVE_PATH, '/', 'SNR_min',           min(SNR_dB_vec));
h5writeatt(SAVE_PATH, '/', 'SNR_max',           max(SNR_dB_vec));
h5writeatt(SAVE_PATH, '/', 'SNR_definition',    'awgn measured over the full fs band (in-band SNR is higher for narrow signals)');
h5writeatt(SAVE_PATH, '/', 'fs_hz',             P.fs);
h5writeatt(SAVE_PATH, '/', 'symbol_rate_hz',    P.Rs);
h5writeatt(SAVE_PATH, '/', 'sps',               int32(P.sps));
h5writeatt(SAVE_PATH, '/', 'seed',              int32(P.seed));
h5writeatt(SAVE_PATH, '/', 'fsk2_sep_hz',       P.fsk2_sep);
h5writeatt(SAVE_PATH, '/', 'fsk4_sep_hz',       P.fsk4_sep);
h5writeatt(SAVE_PATH, '/', 'gfsk_BT',           P.gfsk_BT);
h5writeatt(SAVE_PATH, '/', 'gfsk_pulse_length', int32(P.gfsk_L));
h5writeatt(SAVE_PATH, '/', 'mod_index_gfsk',    P.gfsk_h);
h5writeatt(SAVE_PATH, '/', 'mod_index_cpfsk',   P.cpfsk_h);
h5writeatt(SAVE_PATH, '/', 'USE_PHASE_OFFSET',  int32(P.USE_PHASE_OFFSET));
h5writeatt(SAVE_PATH, '/', 'USE_CFO',           int32(P.USE_CFO));
h5writeatt(SAVE_PATH, '/', 'USE_TIMING_OFFSET', int32(P.USE_TIMING_OFFSET));
h5writeatt(SAVE_PATH, '/', 'USE_FADING',        int32(P.USE_FADING));
h5writeatt(SAVE_PATH, '/', 'USE_RS_VAR',        int32(P.USE_SYMBOL_RATE_VAR));
h5writeatt(SAVE_PATH, '/', 'imp_rows',          'cfo_hz,Rs_scale,timing_offset_samples,k_factor');
h5writeatt(SAVE_PATH, '/', 'USE_PARALLEL',      int32(USE_PARALLEL));
h5writeatt(SAVE_PATH, '/', 'deflate_level',     int32(DEFLATE));
h5writeatt(SAVE_PATH, '/', 'matlab_version',    version);
h5writeatt(SAVE_PATH, '/', 'created',           char(datetime('now')));
h5writeatt(SAVE_PATH, '/', 'complete',          int32(1));   % keep this last - the SCF script checks for it

% Print total time, file size and next step
elapsed = toc(t_start);
fsize   = dir(SAVE_PATH).bytes / 1e9;
fprintf('\nDone. %.1fs  |  File: %s  (%.2f GB)\n', elapsed, SAVE_PATH, fsize);
fprintf('X_iq shape       : [2 x %d x %d]\n', N_samp, N_total);
fprintf('Next             : datasetGenerationSCF_v2.m, with IQ_FILE pointed at this file\n');


%% Local functions

function [iq_frame, imp] = make_iq_frame(mod_type, mod_M, snr, frame_idx, P)
    % Give each frame its own random substream so results are reproducible
    stream = RandStream('mrg32k3a', 'Seed', P.seed);
    stream.Substream = frame_idx;
    RandStream.setGlobalStream(stream);

    N          = P.N_samples;
    sps        = P.sps;
    is_digital = any(strcmp(mod_type, P.has_timing));

    % Generate extra samples so later shifts don't run off the end
    N_work = N + 4*sps + 64;
    N_gen  = ceil(N_work * (1 + P.Rs_var_frac)) + 16;
    tx     = make_signal(mod_type, mod_M, N_gen, P);

    % Randomly speed up or slow down the symbol rate by resampling
    Rs_scale = 1;
    if P.USE_SYMBOL_RATE_VAR && is_digital
        Rs_scale = 1 + (2*rand()-1) * P.Rs_var_frac;
        tx = interp1((0:N_gen-1).', tx, (0:N_work-1).' * Rs_scale, 'spline');   % spline is accurate at 8x oversampling
    else
        tx = tx(1:N_work);
    end

    % Apply a random carrier phase offset
    if P.USE_PHASE_OFFSET
        tx = tx .* exp(1j * 2*pi * rand());
    end

    % Apply a random carrier frequency offset (based on ppm error)
    cfo_hz = 0;
    if P.USE_CFO
        cfo_hz = (2*rand()-1) * P.cfo_ppm_max * 1e-6 * P.f_carrier;
        tx     = tx .* exp(1j * 2*pi * cfo_hz * (0:N_work-1).' / P.fs);
    end

    % Apply a random timing offset of up to one full symbol
    frac_off = 0;
    if P.USE_TIMING_OFFSET && is_digital
        frac_off = rand() * sps;   % 0-1 sample would only cover 1/sps of symbol phases
        n_vec    = (0:N_work-1).';
        tx       = interp1(n_vec, tx, n_vec + frac_off, 'linear', 0);
    end

    % Pass through a Rician fading channel with random K-factor
    K = 0;
    if P.USE_FADING
        K    = P.K_min + rand()*(P.K_max - P.K_min);
        chan = comm.RicianChannel( ...
            'SampleRate',          P.fs, ...
            'PathDelays',          P.path_delays, ...
            'AveragePathGains',    P.path_gains_dB, ...
            'KFactor',             K, ...
            'MaximumDopplerShift', P.doppler_hz, ...
            'RandomStream',        'mt19937ar with seed', ...
            'Seed',                randi(2^31-1));
        d  = info(chan).ChannelFilterDelay;
        tx = chan(tx);
        tx = tx(d+1 : d+N);                 % skip the channel filter delay
        tx = tx / sqrt(mean(abs(tx).^2));   % renormalise to unit power
    else
        tx = tx(1:N);
    end

 % Add AWGN (SNR over full 200 kHz; linear mods ~7.7 dB higher in-band)
    rx = awgn(tx, snr, 'measured');

    % Store as I/Q rows plus the impairment values used
    iq_frame = single([real(rx).'; imag(rx).']);
    imp      = single([cfo_hz; Rs_scale; frac_off; K]);
end

% Generate a clean, unit-power baseband signal for the given modulation
function tx = make_signal(mod_type, mod_M, N, P)
    sps     = P.sps;
    span    = P.span;
    rolloff = P.rolloff;
    fs      = P.fs;
    t       = (0:N-1).' / fs;

    % Linear mods: random symbols, then root-raised-cosine pulse shaping
    switch mod_type
        case {'psk','qam','pam'}
            N_sym   = ceil(N/sps) + span + 10;
            bits    = randi([0 1], N_sym*log2(mod_M), 1);
            sym_idx = bi2de(reshape(bits, N_sym, log2(mod_M)), 'left-msb');
            switch mod_type
                case 'psk'
                    if mod_M == 4
                        syms = pskmod(sym_idx, 4, pi/4);
                    else
                        syms = pskmod(sym_idx, mod_M);
                    end
                case 'qam'
                    syms = qammod(sym_idx, mod_M, 'UnitAveragePower', true);
                case 'pam'
                    syms = complex(pammod(sym_idx, mod_M));
                    syms = syms / sqrt(mean(abs(syms).^2));
            end
            h  = rcosdesign(rolloff, span, sps, 'sqrt');
            tx = filter(h, 1, upsample(syms, sps));
            tx = tx(span*sps/2 + 1:end);
            tx = tx(1:N);

        % GFSK: binary CPM with a Gaussian frequency pulse
        case 'gfsk'
            bits = randi([0 1], ceil(N/sps) + 10, 1);
            obj  = comm.CPMModulator( ...
                'ModulationOrder',      2, ...
                'BitInput',             true, ...
                'FrequencyPulse',       'Gaussian', ...
                'BandwidthTimeProduct', P.gfsk_BT, ...
                'PulseLength',          P.gfsk_L, ...
                'ModulationIndex',      P.gfsk_h, ...
                'SamplesPerSymbol',     sps);
            tx = obj(bits);
            tx = tx(1:N);
     
       % CPFSK: binary continuous-phase FSK from +/-1 input symbols
        case 'cpfsk'
            nrz = 2*randi([0 1], ceil(N/sps) + 10, 1) - 1;   % wants +/-1 in
            obj = comm.CPFSKModulator( ...
                'ModulationOrder',    2, ...
                'ModulationIndex',    P.cpfsk_h, ...
                'SamplesPerSymbol',   sps, ...
                'InitialPhaseOffset', 0);
            tx = obj(nrz);
            tx = tx(1:N);
      
        case 'fsk'
            % FSKModulator has no sample rate setting (it uses SymbolRate x sps).
            % SymbolRate defaults to 100, so without this the tones end up in
            % the wrong place. Set it to 25 kHz so the modulator runs at our 200 kHz.
            N_sym = ceil(N/sps) + 10;
            if mod_M == 2
                idx = randi([0 1], N_sym, 1);
                obj = comm.FSKModulator( ...
                    'ModulationOrder',     2, ...
                    'FrequencySeparation', P.fsk2_sep, ...
                    'SamplesPerSymbol',    sps, ...
                    'SymbolRate',          P.Rs);
            else
                bits = randi([0 1], N_sym*2, 1);
                idx  = bi2de(reshape(bits, N_sym, 2), 'left-msb');
                obj  = comm.FSKModulator( ...
                    'ModulationOrder',     4, ...
                    'FrequencySeparation', P.fsk4_sep, ...
                    'SamplesPerSymbol',    sps, ...
                    'SymbolRate',          P.Rs);
            end
            tx = obj(idx);
            tx = tx(1:N);

     % FM: analog message with 25 kHz peak deviation
        case 'fm'
            msg = analog_message(t);
            obj = comm.FMModulator('SampleRate', fs, 'FrequencyDeviation', 25e3);
            tx  = obj(msg);

        % AM-DSB: carrier plus message at 0.85 modulation index
        case 'am_dsb'
            msg = analog_message(t);
            tx  = complex(1 + 0.85*msg);

        % AM-SSB: analytic signal via Hilbert, middle taken to avoid edges
        case 'am_ssb'
            N_long = 4*N;
            t_long = (0:N_long-1).' / fs;
            msg    = analog_message(t_long);
            ssb    = hilbert(msg);
            mid    = round(N_long/2);
            tx     = ssb(mid:mid+N-1);

        otherwise
            error('Unknown mod_type: %s', mod_type);
    end
    tx = tx / sqrt(mean(abs(tx).^2));   % normalise to unit power
end

% Fake audio: 8 random tones under 8 kHz, scaled to +/-1
function msg = analog_message(t)
    freqs  = rand(8,1) * 8e3;
    phases = rand(8,1) * 2*pi;
    amps   = rand(8,1);
    msg    = sum(amps.' .* cos(2*pi*freqs.' .* t + phases.'), 2);
    msg    = msg / max(abs(msg));
end

%% Self-test
% Ensure the signals are being generated as intended

function all_ok = run_iq_selftest(P, mod_table)
    fprintf('=== I/Q SELF-TEST ===\n');
    fs     = P.fs;
    Rs     = P.Rs;
    % cap the test record so the self-test still takes seconds. 32768
    % samples is plenty to see the tones and the symbol-rate line
    N_long = min(8 * P.N_samples, 32768);
    n_fail = 0;

    function report(name, value, ok, expect)
        if ok, tag = 'PASS'; else, tag = 'FAIL'; n_fail = n_fail + 1; end
        fprintf('  [%s] %-50s %9.2f   (%s)\n', tag, name, value, expect);
    end

    % FSK tones land where we asked. use instantaneous freq (phase step
    % per sample) since for CP-FSK every sample sits exactly on a tone
    for kind = {2, P.fsk2_sep; 4, P.fsk4_sep}.'
        M   = kind{1};
        sep = kind{2};
        x      = make_signal('fsk', M, N_long, P);
        f_inst = angle(x(2:end) .* conj(x(1:end-1))) * fs / (2*pi);
        tones  = unique(round(f_inst));
        want   = ((0:M-1) - (M-1)/2) * sep;
        if numel(tones) == M
            v = max(abs(tones(:).' - want));
        else
            v = inf;
        end
        report(sprintf('%dFSK tones, worst error vs wanted (Hz)', M), v, v < 1, ...
               [mat2str(want/1e3) ' kHz']);
    end

    % Ensure 2FSK isn't just a copy of CPFSK
    x2 = make_signal('fsk',   2, 4096, P);
    xc = make_signal('cpfsk', 2, 4096, P);
    v  = median(abs(angle(x2(2:end) .* conj(x2(1:end-1))))) / ...
         median(abs(angle(xc(2:end) .* conj(xc(1:end-1)))));
    report('2FSK / CPFSK phase step per sample', v, abs(v - 1) > 0.5, ...
           'expect 2 (h = 1 vs 0.5)');

    % Same for GFSK vs CPFSK. For L = 1 GFSK is just MSK (= CPFSK at h = 0.5).
    % L = 4 is way tighter in freq, so occupied BW should split them
    xg = make_signal('gfsk',  2, N_long, P);
    xc = make_signal('cpfsk', 2, N_long, P);
    v  = occ_bw(xg, fs, 0.95) / occ_bw(xc, fs, 0.95);
    report('GFSK / CPFSK 95% occupied bandwidth', v, v < 0.9, ...
           'expect ~0.5, near 1 means GFSK = MSK');

    % Check the jitter actually moves the symbol rate.
    % |x|^2 of pulse-shaped BPSK has a spectral line at Rs,
    % so just find it
    Ng = ceil(N_long * 1.03) + 64;
    x  = make_signal('psk', 2, Ng, P);
    for scale = [1 1.02]
        xj = interp1((0:Ng-1).', x, (0:N_long-1).' * scale, 'spline');
        e  = abs(xj).^2;
        E  = abs(fft(e - mean(e)));
        fa = (0:N_long-1) * fs / N_long;
        band   = find(fa > 0.8*Rs & fa < 1.2*Rs);
        [~, k] = max(E(band));
        v = fa(band(k));
        report(sprintf('symbol-rate line with jitter x%.2f (Hz)', scale), v, ...
               abs(v - scale*Rs) <= 2*fs/N_long, sprintf('expect %.0f', scale*Rs));
    end

    % full pipeline on every class and check everything is working properly:
    % right size, no NaN/Inf, power = 1, logged impairments in range,
    % and the same frame index gives the exact same frame twice for
    % reproducibility
    ok = true;
    for m = 1:size(mod_table, 1)
        [iq,  imp1] = make_iq_frame(mod_table{m,3}, mod_table{m,4}, 30, 7 + m, P);
        [iq2, imp2] = make_iq_frame(mod_table{m,3}, mod_table{m,4}, 30, 7 + m, P);
        p       = mean(double(iq(1,:)).^2 + double(iq(2,:)).^2);
        digital = any(strcmp(mod_table{m,3}, P.has_timing));
        in_range = abs(imp1(1)) <= P.cfo_ppm_max*1e-6*P.f_carrier ...
            && imp1(4) >= P.K_min && imp1(4) <= P.K_max ...
            && ((digital && abs(imp1(2) - 1) <= P.Rs_var_frac && imp1(3) >= 0 && imp1(3) < P.sps) ...
                || (~digital && imp1(2) == 1 && imp1(3) == 0));
        ok = ok && isequal(size(iq), [2 P.N_samples]) && all(isfinite(iq(:))) ...
             && abs(p - 1) < 0.05 && in_range && isequal(iq, iq2) && isequal(imp1, imp2);
    end
    report('pipeline: every class sized, finite, power ~1, repeatable', double(ok), ok, '1 = yes');

    all_ok = (n_fail == 0);
    if all_ok
        fprintf('=== ALL PASSED ===\n');
    else
        fprintf('=== %d CHECK(S) FAILED - fix before generating ===\n', n_fail);
    end
end

function bw = occ_bw(x, fs, frac)
    % works out how wide a signal is in frequency.
    % It's only used in test 4 to show GFSK is narrower than CPFS
    % width of the band holding 'frac' of the power (e.g. 0.95 = middle 95%).
    % Hann window first, otherwise FFT leakage makes every signal look wider
    N   = numel(x);
    w   = 0.5 - 0.5*cos(2*pi*(0:N-1).' / N);
    Pxx = fftshift(abs(fft(x .* w)).^2);
    f   = (-N/2 : N/2-1).' * fs / N;
    c   = cumsum(Pxx) / sum(Pxx);
    lo  = f(find(c >= (1-frac)/2,     1));
    hi  = f(find(c >= 1 - (1-frac)/2, 1));
    bw  = hi - lo;
end
