function model = load_model(filename)
fid = fopen(filename,'rb');

% Read in the encoder parameters
model.num_classifiers = fread(fid,1,'int');
model.f = zeros(model.num_classifiers,1);
model.t = zeros(model.num_classifiers,1);
model.e = zeros(model.num_classifiers,1);
for i = 1:model.num_classifiers
    model.f(i) = fread(fid,1,'int');
    model.t(i) = fread(fid,1,'unsigned char');
    model.e(i) = fread(fid,1,'double');
end

% Read in the decoder parameters
model.num_bits = fread(fid,1,'int'); % Number of bits/number of classifiers
model.num_dict = fread(fid,1,'int'); % Dictionary size
model.num_bits_max = 30; % Magic number for now
model.dec = dictionary([],{});
for i = 1:model.num_dict
    c = 0;
    for j = 1:model.num_bits
        a = fread(fid,1,'char');
        if(a == '1')
            c = c + 2^(j-1);
        end
    end
    model.num_classes = fread(fid,1,'int');
    prob = fread(fid,model.num_classes,'int');
    num = fread(fid,1,'int');
    label = fread(fid,1,'unsigned char');
    model.dec{c} = {prob, num, label};
end
fclose(fid);