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
        'select': 'multi'
    });

    $('#submit').prop('disabled', true);

    let selected = 0;

    $('#target tbody').on('click', 'tr', function () {
        if ($(this).hasClass('selected')) {
            selected -= 1;
            if (selected === 0) {
                $('#submit').prop('disabled', true);
            }
        }
        else {
            $(this).addClass('selected');
            selected += 1;
            if (selected > 0) {
                $('#submit').prop('disabled', false);
            }
        }
    });

    $('form').submit(function () {
        let selected_row = table.row('.selected').data();
        var input = $("<input>")
            .attr("type", "hidden")
            .attr("name", "selected_row").val(JSON.stringify(selected_row));
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
