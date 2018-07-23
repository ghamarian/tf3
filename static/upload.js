function add_new_row(dataset_selected, model, acc, loss) {
    $('#tablediv').append('<tr>');
    $('#tablediv').append('<td> <input type="radio" id="exisiting_files-configuration" name="exisiting_files-configuration" value="' + dataset_selected + '">' + dataset_selected);
    $('#tablediv').append('</td> <td> ' + model + ' </td> <td> ' + acc + '</td> <td> ' + loss + '</td> </tr>');
};
