classdef cvvdp
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        conda_env
    end

    methods
        function obj = cvvdp(conda_env)
            obj.conda_env = conda_env;
        end

        function jod = cmp(obj, img_test, img_ref, display)
            arguments
                obj
                img_test {mustBeReal}
                img_ref {mustBeReal}
                display = 'standard_4k'
            end
            
            test_file = strcat( tempname(), '.png' );
            ref_file = strcat( tempname(), '.png' );

            imwrite( img_test, test_file );
            imwrite( img_ref, ref_file );
            
            cmd = [ 'conda activate ', obj.conda_env, '; cvvdp --test "', test_file, '" --ref "', ref_file, '" --display ', display, ' --quiet' ];

            if ispc()
                cmd = [ '"%PROGRAMFILES%\Git\bin\sh.exe" -l -c ''', cmd, '''' ];
            end

            [status, cmdout] = system( cmd );
            if status ~= 0
                error( 'cvvdp: Something went wrong:\n %s\n', cmdout )
            else
                jod = str2double(cmdout);
            end

            delete( test_file );
            delete( ref_file );           

        end
    end
end