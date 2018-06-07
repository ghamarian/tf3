var CATEGORIES = CATEGORIES || (function () {
    let _args = {}; // private

    return {
        init: function (Args) {
            _args = Args;
        },

        get: function (cat) {
            return _args[cat];
        }
    };
}());

$(document).ready(function () {
    let categorical = 'categorical';
    let hash = 'hash';
    let none = 'none';
    let range = 'range';
    let numerical = 'numerical';

    let options = {
        numerical: () => $('<option>').attr('value', numerical).text('Numerical'),
        categorical: () => $('<option>').attr('value', categorical).text('Categorical'),
        hash: () => $('<option>').attr('value', hash).text('Hash'),
        none: () => $('<option>').attr('value', none).text('No use'),
        range: () => $('<option>').attr('value', range).text('Range')
    };

    function createMenu(selected, ...items) {
        let result = $("<select>");
        for (let i = 0; i < items.length; i++) {
            result.append(options[items[i]]());
        }
        result.find('option[value=' + selected + ']').attr('selected', true);
        return result.prop('outerHTML');
    }


    let category = {
        'categorical': createMenu(categorical, categorical, hash, none),
        'hash': createMenu(hash, hash, categorical, none),
        'int-range': createMenu(range, range, hash, categorical, numerical, none),
        'int-hash': createMenu(hash, hash, categorical, numerical, none),
        'int-category': createMenu(hash, hash, range, categorical, numerical, none),
        'bool': createMenu(categorical, categorical, none),
        'numerical': createMenu(numerical, numerical, none)
    };

    table_tag = $('#amir');

    table = table_tag.DataTable({
        "columnDefs": [
            {
                "render": function (data, type, row) {
                    return category[data];
                },
                "targets": 1
            },
            {
                "render": function (data, type, row) {
                    return data == -1 ? 'Not relevant' : data;
                },
                "targets": 2
            },
            {
                "render": function (data, type, row) {
                    return '<input type="text" name=data value=' + data + '>'
                },
                "targets": 3
            }
        ],
        'ordering': false,
        // "pageLength": 2
    });

    // console.log($('#category-data').data());

    $('form').submit(function () {
        let cat_column = table.$('select option:selected').map(function() {
            return this.value;
        }).get();

        let default_column = table.$('input').map(function() {
            return this.value;
        }).get();

        console.log(default_column);

        let input = $("<input>")
            .attr("type", "hidden")
            .attr("name", "cat_column").val(JSON.stringify(cat_column));
        $('form').append($(input));
    });

});
