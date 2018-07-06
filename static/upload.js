  $(document).ready(function () {
            var handle_key = {};
            handle_key.configs = {{ user_configs | tojson | safe }};
            handle_key.parameters = {{ parameters | tojson | safe }};
            var dataset_selected = $('#exisiting_files-train_file_exist').find("option:selected").text();
            var len = handle_key.configs[dataset_selected].length;
            $('#exisiting_files-configuration').find('option').remove();
            var i;
            for (i = 0; i < len; i++) {
                var data_and_conf = dataset_selected + '_' + handle_key.configs[dataset_selected][i];
                var handle_key2 = {};
                handle_key2.configs_parame = handle_key.parameters[data_and_conf];
                $('#tablediv').append('<tr>');
                $('#tablediv').append('<td> <input type="radio" id="exisiting_files-configuration" name="exisiting_files-configuration" value="' + handle_key.configs[dataset_selected][i] + '">' + handle_key.configs[dataset_selected][i]);
                $('#tablediv').append('<td> ' + handle_key2.configs_parame['model']);
                $('#tablediv').append('<td> ' + handle_key2.configs_parame['acc']);
                $('#tablediv').append('<td> ' + handle_key2.configs_parame['loss']);
                $('#tablediv').append('</tr>');

            }
            $('#exisiting_files-train_file_exist').change(function () {
                var dataset_selected = $('#exisiting_files-train_file_exist').find("option:selected").text();
                var len = handle_key.configs[dataset_selected].length;

                $('#exisiting_files-configuration').find('option').remove();
                $("#tablediv").empty();
                var i;
                for (i = 0; i < len; i++) {
                    var data_and_conf = dataset_selected + '_' + handle_key.configs[dataset_selected][i];
                    var data_and_conf = dataset_selected + '_' + handle_key.configs[dataset_selected][i];
                    var handle_key2 = {};
                    handle_key2.configs_parame = handle_key.parameters[data_and_conf];
                    $('#tablediv').append('<tr>');
                    $('#tablediv').append('<td> <input type="radio" id="exisiting_files-configuration" name="exisiting_files-configuration" value="' + handle_key.configs[dataset_selected][i] + '">' + handle_key.configs[dataset_selected][i]);
                    $('#tablediv').append('<td> ' + handle_key2.configs_parame['model']);
                    $('#tablediv').append('<td> ' + handle_key2.configs_parame['acc']);
                    $('#tablediv').append('<td> ' + handle_key2.configs_parame['loss']);
                    $('#tablediv').append('</tr>');

                }
            });
            $('#tablediv').on('click', 'tr', function () {
                if ($(this).hasClass('selected')) {
                    $(this).removeClass('selected');
                    $('#submit').prop('disabled', true);
                }
                else {
                    table.$('tr.selected').removeClass('selected');
                    $(this).addClass('selected');
                    $('#submit').prop('disabled', false);
                }
            });
        });