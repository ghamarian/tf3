$(document).ready(function () {
    table_tag = $('#target');
    table = table_tag.DataTable({
        "columnDefs": [
            {
                "render": function (data, type, row) {
                    return data == -1 ? 'Not relevant' : data;
                },
                // "targets": 2
            }
        ],
        'ordering': false,
        'select': 'api'
    });

    $('#submit').prop('disabled', true);

    $('#target tbody').on('click', 'tr', function () {
        if (table.row(this, {selected: true}).any()) {
            table.row(this).deselect();
        }
        else {
            table.row(this).select();
        }

        if (table.rows({selected: true}).any()) {
            $('#submit').prop('disabled', false);
        }
        else {
            $('#submit').prop('disabled', true);
        }
    });

    $('form').submit(function () {
        // let selected_rows = table.rows({selected: true}).data().map
        let selected_rows = [];
        table.rows({selected: true}).every(function (rowIdx, tableLoop, rowLoop) {
            selected_rows.push(this.data()[0]);
        });
        let input = $("<input>")
            .attr("type", "hidden")
            .attr("name", "selected_rows").val(JSON.stringify(selected_rows));
        $('form').append($(input));
    });

});

// function add_selected_target(target_selected) {
//     // add_selected_target($('#target tr:contains({{ target_selected }})')).addClass('selected');
//         target_selected.addClass('selected');
//     if ($("#target tr").hasClass("selected")) {
//         $("input").prop('disabled', false);
//     }
//
// };
//
