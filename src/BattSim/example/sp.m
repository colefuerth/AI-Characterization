inp = input('', 's');
inp = sscanf(inp, '%f');
if length(inp) ~= 8
    error('Invalid input');
end
disp(inp);