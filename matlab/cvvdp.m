classdef cvvdp
    % A wrapper class to run ColourVideoVDP from Matlab
    %
    % Example:
    % v = cvvdp( 'cvvdp' ); % the string must be a name of conda
    %                       % environment with installed cvvdp
    % img_ref = imread( '../example_media/wavy_facade.png' );
    % img_test = imnoise( img_ref, 'gaussian', 0, 0.001 );
    % v.cmp( img_test, img_ref, 'standard_fhd' )
    %
    % The arguments and options passed to this class reflect those that are
    % passed to `cvvdp` command line - check `cvvdp --help` for more information on those. 

    properties
        conda_env
        device = []
    end

    methods
        function obj = cvvdp(conda_env, device) 
            % conda_env - the name of the conda environment with installed
            %             cvvdp
            % device - device to run cvvdp on: 'cpu', 'mps', 'cuda:0', 'cuda:1', ...

            obj.conda_env = conda_env;
            if exist( 'device', 'var' )
                obj.device = device;
            end
        end

        function [jod, heatmap] = cmp(obj, img_test, img_ref, display, options )
            arguments
                obj
                img_test {mustBeReal}
                img_ref {mustBeReal}
                display = 'standard_4k'
                options.fps (1,1) {mustBePositive} = 30
                options.ppd (1,1) {mustBeNumeric} = -1
                options.heatmap {mustBeMember(options.heatmap, {'none', 'raw', 'threshold', 'supra-threshold'})} = 'none'
                options.verbose (1,1) = false
                options.config_paths {mustBeText} = ''
            end
            
            if isa( img_test, 'double' )
                img_test = single(img_test);
            end
            if isa( img_ref, 'double' )
                img_ref = single(img_ref);
            end

            test_file = strcat( tempname(), '.mat' );
            ref_file = strcat( tempname(), '.mat' );

            ppd = options.ppd;
            fps = options.fps;
            save( test_file, 'img_test', 'fps' )
            save( ref_file, 'img_ref', 'fps' )
            %imwrite( img_test, test_file );
            %imwrite( img_ref, ref_file );
            
            if ppd>0
                ppd_arg = sprintf( ' --pix-per-deg %g', ppd );
            else
                ppd_arg = '';
            end

            if ~strcmp(options.heatmap, 'none')
                tmp_dir = fileparts( test_file );
                heatmap_arg = [ ' --heatmap ', options.heatmap, ' --output-dir "', strrep(tmp_dir, '\', '/'), '"' ];
            else
                heatmap_arg = '';
            end                
            
            cmd = [ 'conda activate ', obj.conda_env, '; cvvdp --test "', test_file, '" --ref "', ref_file, '" --display ', display, ppd_arg, heatmap_arg ]; 
%             if ~options.verbose
%                 cmd = [cmd, ' --quiet'];
%             end
            if ~isempty( options.config_paths )
                cmd = [cmd, ' --config-paths "', options.config_paths, '"'];
            end
            if ~isempty(obj.device)
                cmd = [cmd, ' --device ', obj.device];
            end

            if ispc()
                cmd = [ '"%PROGRAMFILES%\Git\bin\sh.exe" -l -c ''', cmd, '''' ];
            elseif ismac()
                cmd = [ '/bin/bash -l -c ''', cmd, '''' ];
            else
                cmd = [ '/usr/bin/bash -l -c ''', cmd, '''' ];
            end

            [status, cmdout] = system( cmd );
            if status ~= 0
                error( 'cvvdp: Something went wrong:\n %s\n', cmdout )
            else
                if options.verbose
                    fwrite( 2, cmdout );
                end
                jod_res = regexp( cmdout, "cvvdp=[\d\.]*", "match" );
                jod = str2double(jod_res{1}(7:end));
            end

            if ~strcmp(options.heatmap, 'none')
                heatmap_fn = [ test_file(1:(end-4)), '_heatmap.png' ];
                if ~isfile( heatmap_fn )
                    warning( 'cvvdp: Missing heatmap files - something went wrong' )
                    heatmap = [];
                else
                    heatmap = imread( heatmap_fn );
                    delete( heatmap_fn );
                end
            else
                heatmap = [];
            end

            delete( test_file );
            delete( ref_file );           

        end
    end
end