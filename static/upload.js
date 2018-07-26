// function add_new_row(dataset_selected, model, acc, loss) {
//     $('#tablediv').append('<tr> ');
//     $('#tablediv').append('<td> <input type="radio" id="exisiting_files-configuration" name="exisiting_files-configuration" value="' + dataset_selected + '">' + dataset_selected);
//     $('#tablediv').append('</td> <td> ' + model + ' </td> <td> ' + acc + '</td> <td> ' + loss + '</td> ');
//     $('#tablediv').append('<td> <a data-id=' + dataset_selected + ' onclick="ConfirmDelete(this, false)" ><span class="glyphicon glyphicon-remove"></span></a>');
//     $('#tablediv').append(' </tr>');
// };

function add_new_row(dataset_selected, model, acc, loss) {
  var row = '<tr> <td> <input type="radio" id="exisiting_files-configuration" name="exisiting_files-configuration" value="' + dataset_selected +
      '">' + dataset_selected +'</td> <td> ' + model + ' </td> <td> ' + acc + '</td> <td> ' + loss + '</td> <td> <a data-id=' + dataset_selected +
      ' onclick="ConfirmDelete(this, false)" ><span class="glyphicon glyphicon-remove"></span></a> </tr>';
    $("#table_config tbody").append(row);
};

