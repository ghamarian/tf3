
$(document).ready(function () {

    table_tag = $('#target');

    table = table_tag.DataTable({
        "columnDefs": [
            {
                "render": function (data, type, row) {
                    return data === -1 ? 'Not relevant' : data;
                },
                "targets": 2
            }
        ],
        'ordering': false,
        'select': true
        // "pageLength": 2
    });

    table_tag.on('page.dt', function () {
        let info = table.page.info();
        $('#pageInfo').html('Showing page: ' + info.page + ' of ' + info.pages);
        let data = table.$('select option:selected').text();
    });

    $('#select').click(function (event) {
        event.preventDefault();
        // var cat_column = table.columns(1).data()[0]
        let cat_column = table.$('select option:selected').text().split();


        $.ajax('/cat_col', {
            data: {'cat_column': JSON.stringify(cat_column)},
            dataType: 'json',
            type: "POST",
            // contentType: "application/json; charset=utf-8",
            cache: false,
            // data: {"a": "b"},
            success: function (data) {
                $('#next').prop('disabled', false);
            }
        });
    });


})
;
