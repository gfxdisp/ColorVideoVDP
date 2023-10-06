classdef cvvdp
    % A wrapper class to run ColourVideoVDP from matlab
    %
    % Example:
    % v = cvvdp( 'cvvdp' ); % the string must be a name of conda
    %                       % environment with installed cvvdp
    % img_ref = imread( '../example_media/wavy_facade.png' );
    % img_test = imnoise( img_ref, 'gaussian', 0, 0.001 );
    % v.cmp( img_test, img_ref, 'standard_fhd' )

    properties
        conda_env
    end

    methods
        function obj = cvvdp(conda_env)
            obj.conda_env = conda_env;
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
                heatmap_arg = [ ' --heatmap ', options.heatmap, ' --output-dir ', strrep(tmp_dir, '\', '/') ];
            else
                heatmap_arg = '';
            end
                

            cmd = [ 'conda activate ', obj.conda_env, '; cvvdp --test "', test_file, '" --ref "', ref_file, '" --display ', display, ppd_arg, heatmap_arg, ' --quiet' ]; 

            if ispc()
                cmd = [ '"%PROGRAMFILES%\Git\bin\sh.exe" -l -c ''', cmd, '''' ];
            else
                cmd = [ '/usr/bin/bash -l -c ''', cmd, '''' ];
            end

            [status, cmdout] = system( cmd );
            if status ~= 0
                error( 'cvvdp: Something went wrong:\n %s\n', cmdout )
            else
                jod = str2double(cmdout);
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
            end

            delete( test_file );
            delete( ref_file );           

        end
    end
end