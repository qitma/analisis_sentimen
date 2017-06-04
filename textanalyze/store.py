import json,csv
import xlsxwriter
import xlrd

class Store(object):
    def import_profil_to_excel(cls,header_format,filename,data,sheet_name):
        """
        :param header_format: berupa list contoh : [1,2,3]
        :param filename: nama file
        :param data: berupa dictionary dengan aturan value berupa list contoh key:[1,2,3]
        :param sheet_name: nama dari sheet nya
        :return:
        """
        workbook = xlsxwriter.Workbook(filename=filename+".xlsx")
        profil_sheet = workbook.add_worksheet(sheet_name)

        c = 0
        for header in header_format:
            profil_sheet.write(0,c,header)
            c+=1

        row = 1
        col = 0
        for trait,data_col in data.items():
            profil_sheet.write(row,col,trait)
            for count_col in range(1,len(data_col)+1):
                profil_sheet.write(row,count_col,data_col[count_col-1])
            row+=1

        workbook.close()
        #trick biar bisa edit di file yang sama...
        wbRD = xlrd.open_workbook("{}.xlsx".format(filename))
        sheets = wbRD.sheets()

        workbook = xlsxwriter.Workbook(filename=filename + ".xlsx")

        for sheet in sheets:  # write data from old file
            newSheet = workbook.add_worksheet(sheet.name)
            for row in range(sheet.nrows):
                for col in range(sheet.ncols):
                    newSheet.write(row, col, sheet.cell(row, col).value)
        row = 1
        col = 0
        for trait,data_col in data.items():
            newSheet.write(row,col,trait)
            for count_col in range(1,len(data_col)+1):
                newSheet.write(row,count_col,data_col[count_col-1])
            row+=1
        print("Import profile to excel succes...")

    def import_profil_to_excel_batch(cls, header_format, filename, batch_data, worksheet_name, fold_number):
        """
        :param header_format: berupa list contoh : [1,2,3]
        :param filename: nama file
        :param batch_data: berupa list of dictionary dengan aturan value berupa list contoh data =  [key:[1,2,3],key2:[1,2,3]]
        :param worksheet: nama dari sheet nya
        :return:
        """
        workbook = xlsxwriter.Workbook(filename=filename+".xlsx")
        worksheet = workbook.add_worksheet(worksheet_name)
        merge_format1 = workbook.add_format({
            'align': 'left',
            'valign': 'vcenter',
            'fg_color': 'yellow'})
        merge_format2 = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#92D050'})
        worksheet.merge_range('A2:F3', 'K-FOLD'+str(fold_number), merge_format1)
        awal_col = 0
        batas_col = 3
        count = 1
        length_col = 0
        for data in batch_data:
            row = 3
            worksheet.merge_range(row,awal_col,row,batas_col, 'Iterasi' + str(count), merge_format2)
            row_header = row +1
            c = awal_col
            for header in header_format:
                worksheet.write(row_header, c, header)
                c+=1
            row_item = row_header + 1
            for trait,data_col in data.items():
                worksheet.write(row_item, awal_col, trait) #bener
                length_col = len(data_col) #=3
                for count_col in range(1,length_col+1):
                    worksheet.write(row_item, awal_col+count_col, data_col[count_col - 1])
                row_item+=1
            awal_col += length_col+2
            batas_col += length_col+2
            count+=1

        workbook.close()


        print("Import profile to excel succes...")

    def import_performa_batch_al(cls, header_format, filename, batch_data, worksheet_name, iteration_number):
        """
        :param header_format: berupa list contoh : [1,2,3]
        :param filename: nama file
        :param batch_data: berupa list of dictionary dengan aturan value berupa list contoh data =  [key:[1,2,3],key2:[1,2,3]]
        :param worksheet: nama dari sheet nya
        :return:
        """
        workbook = xlsxwriter.Workbook(filename=filename + ".xlsx")
        worksheet = workbook.add_worksheet(worksheet_name)
        merge_format1 = workbook.add_format({
            'align': 'left',
            'valign': 'vcenter',
            'fg_color': 'yellow'})
        merge_format2 = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#92D050'})
        worksheet.merge_range('A2:F3', 'Iterasi' + str(iteration_number), merge_format1)
        awal_col = 0
        batas_col = 3
        count = 1
        length_col = 0
        for data in batch_data:
            row = 3
            worksheet.merge_range(row, awal_col, row, batas_col, 'Query ke ' + str(count), merge_format2)
            row_header = row + 1
            c = awal_col
            for header in header_format:
                worksheet.write(row_header, c, header)
                c += 1
            row_item = row_header + 1
            for trait, data_col in data.items():
                worksheet.write(row_item, awal_col, trait)  # bener
                length_col = len(data_col)  # =3
                for count_col in range(1, length_col + 1):
                    worksheet.write(row_item, awal_col + count_col, data_col[count_col - 1])
                row_item += 1
            awal_col += length_col + 2
            batas_col += length_col + 2
            count += 1

        workbook.close()

        print("Import profile to excel succes...")

